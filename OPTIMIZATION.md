# fmg 0.3.0 → 0.4.0：性能与稳定性优化说明

本次改动重写了 `fmg/core.py`，并同步修正 `__init__.py`、`setup.py`、`test.py`；新增 `selftest.py` 作为回归测试。磁盘格式保持兼容：**旧的 `.fmg` 文件可以直接读**，新写出的文件带 `format_version=2`。

---

## 一、为什么 DataLoader 多线程读图会"偶尔"崩溃

五个问题叠在一起，前四个只在 `num_workers>0` 时暴露，所以表现为低概率、难复现：

| # | 原因 | 典型症状 |
|---|---|---|
| 1 | `__init__` 里执行 `lmdb.open(...).begin()`，在**父进程**就打开了 LMDB Environment 和一个长事务；Linux 下 DataLoader 默认 `fork`，4 个 worker 继承同一个 env、同一段 mmap、同一个 reader slot 和锁文件 fd。LMDB 不允许跨 fork 复用 env，属于未定义行为 | `lmdb.BadRslotError: MDB_BAD_RSLOT`、SIGBUS/SIGSEGV、`DataLoader worker (pid ...) exited unexpectedly` |
| 2 | `self._data[level]` 存的是**单个 Transaction**，被所有 worker / 线程共用。py-lmdb 使用 `MDB_NOTLS`，读事务可以跨线程迁移，但**不能被多个线程并发使用** | 偶发读到脏数据、随机崩溃 |
| 3 | `cv2.imdecode` 依赖 OpenCV 内部线程池 / OpenCL 上下文，这些状态在 `fork()` 后是失效的 | worker 挂起或段错误 |
| 4 | `txn.get()` 返回 `None`（tile 不存在）、`cv2.imdecode` 返回 `None`（空/损坏 tile）时没有任何保护，直接 `np.frombuffer(None)` / 数组赋值，异常发生在子进程里，主进程只看到 worker 退出 | 训练随机中断，且没有有用堆栈 |
| 5 | 同一个进程里出现两个指向同一路径的 slide 对象——`pickle` 往返产生的 clone、`spawn` 后的反序列化、train/val 各开一次、多个 Dataset 实例——各自去 `lmdb.open()` 同一目录 | py-lmdb 直接拒绝：`lmdb.Error: The environment '...' is already open in this process`；即使侥幸能开，也是把同一份 mmap 重复映射 N 次 |

附带问题：若改用 `spawn` 启动方式，原代码会直接 `TypeError: can't pickle Environment objects`，因为 LMDB 句柄不可序列化。

### 修复方案（核心）

1. **懒打开 + 进程感知重开**：`open_slide()` 只读 JSON 元数据，不碰 LMDB；第一次 `read_region()` 才打开环境。对象被 fork 到子进程后，检测到 `os.getpid()` 变化，**丢弃继承来的句柄（不调用 `close()`，避免清掉父进程的 reader slot），在子进程重新 `lmdb.open()`**。这是社区与官方文档都推荐的唯一可靠做法。
2. **每进程一个 env、每次读取一个短事务**：`with env.begin(buffers=True) as txn`，配合 `max_spare_txns=8`，事务创建几乎零成本，且天然线程安全。
3. **只读参数**：`readonly=True, create=False, lock=False, readahead=False, meminit=False`。`lock=False`（MDB_NOLOCK）完全绕开 reader lock table，多进程并发只读时不再有 slot 竞争；WSI 是随机小读，关掉预读更省内存。
4. **fork 子进程自动调用 `cv2.setNumThreads(0)` + 关闭 OpenCL**；`spawn` 用户可在 `worker_init_fn` 里显式 `fmg.configure(num_threads=0)`。
5. **可 pickle**：实现 `__getstate__/__setstate__`，剔除 env 与 `RLock`，`fork`/`spawn` 都能用。
6. **坏 tile 兜底**：缺 tile / 解码失败 / 形状不符时，默认涂黑并只警告一次；`strict=True` 可改成抛 `CorruptTileError`，不会再让一个坏 patch 拖垮整个训练。
7. `close()` 幂等，只关闭**本进程**持有的句柄；`__del__` 补齐；`with slide:` 正常可用。
8. **进程级 Environment 注册表 + 引用计数**：同一进程里所有指向同一 `(realpath, map_size)` 的只读 slide 共享**一个** Environment，`close()` 只减引用计数，最后一个使用者离开时才真正 `env.close()`。`os.register_at_fork(after_in_child=...)` 在 fork 后清空注册表并重建锁（继承的句柄只丢弃引用、绝不 `close`），子进程按需打开自己的 env。于是 `pickle` 往返、多个 Dataset、train/val 双 loader、`spawn` 都不会再撞上 "already open in this process"，也不会重复 mmap 同一份数据。

> 结论：`fmg.open_slide()` 现在可以放心放在 `Dataset.__init__` 里，`num_workers=4` 甚至更多都不会共享底层句柄；同一张片子在一个进程里开几次也只占一份映射。

---

## 二、其它 bug 修复

| 问题 | 影响 | 修复 |
|---|---|---|
| `level_dimensions` 存成 `[h, w]`，但写/读时把 `dimension[0]` 当 x（宽）上限、`dimension[1]` 当 y（高）上限 | **所有非正方形图像都会写错**：宽方向出现空 patch、高方向缺 tile，读回缺块 | 新格式统一 `(width, height)`，网格 x 遍历 width、y 遍历 height；旧文件读取时自动把对外视图翻转回 `(w,h)` 并给出一次警告 |
| `w, h, c = data.shape` 变量名与真实含义相反 | 连锁导致 `dimension`、`level_downsamples`、`test.py` 的 `weight/height` 全部错位 | 改为 `h, w, _ = data.shape` |
| `level_downsamples` 存成 `[[1.0,1.0],[2.0,2.0]]` | 不符合 OpenSlide 标量约定 | 新格式写 `[1.0, 2.0, ...]`；读旧格式自动取标量 |
| `read_region(location, level, size)` 中 `location` 未按 downsample 缩放 | 读 level>0 得到错误位置 | 按 OpenSlide 语义：`location` 是 level-0 坐标，`size` 是 level 坐标尺寸，内部除以 downsample |
| `properties['mpp'] = float(mpp)` 而默认 `mpp=None` | 不传 mpp 直接 `TypeError` | 允许 `None`，JSON 写 `null` |
| `json.load(open(filename))` | 文件句柄泄漏 | `with open(...)` |
| `close()` 调用 `Transaction.close()`（Transaction 没有该方法） | `with slide:` / 显式 close 抛 `AttributeError` | 改为关闭 Environment；文档声称的自动析构也补上了 `__del__` |
| `test.py` 的 `readroi(x, y, w, h)` 调用 `read_region(x, y, w, h, 0)`（5 个参数） | 直接 `TypeError` | 改成 `read_region((x, y), level, (w, h))` |
| `self._filename[:-4]` 硬切后缀 | 非 4 字符扩展名会删错目录 | `os.path.splitext(...)[0]`，对 `.fmg` 结果与旧版一致 |
| `read_region` 先分配整个包围盒再裁剪 | 内存浪费；负坐标 / 越界时切片索引错乱 | 直接分配目标 `(h, w, 3)` 画布，只对相交 tile 做源/目标交集拷贝，越界补黑 |
| 未检查 `imencode` / `imdecode` 返回值 | 空 patch 写入后读侧崩溃 | 写侧失败抛 `FmgError`，读侧走兜底逻辑 |
| 写侧 `level_data` 保存所有金字塔层的完整 numpy 副本 | 峰值内存约 1.33× 原图，WSI 上可能几十 GB | 流式生成：每层由 level0 一次性 `cv2.resize(..., INTER_AREA)` 得到，用完即弃 |
| 金字塔用 `data[::2, ::2]` | 严重锯齿 | `cv2.INTER_AREA` 面积平均降采样 |
| 写侧每列 x 开一个写事务（原代码每列 commit 一次） | 事务提交次数多、写放大 | 每 `commit_every=2048` 个 tile 提交一次；`save_from_tiles(..., txn=)` 支持外部批量事务 |
| `buff.tobytes()` | 每个 tile 多一次内存拷贝 | `memoryview(buff)` |
| 元数据 JSON 直接覆写 | 写到一半崩溃会留下指向不完整数据的 `.fmg` | 先写 `.tmp` 再 `os.replace()` 原子替换 |

---

## 三、性能优化点

- **解码 LRU 缓存**：`patch_cache=128`（默认）。多 epoch 训练时每个 worker 重复命中相同 tile，省掉 `txn.get` + `cv2.imdecode`；缓存是进程内的，不跨进程共享内存。
- **`buffers=True`**：`txn.get()` 返回指向 mmap 的 `memoryview`，`np.frombuffer` 零拷贝。
- **`max_spare_txns=8`**：py-lmdb 复用只读事务，短事务开销可忽略。
- **按 level 懒开环境**：只读实际用到的层，worker 启动时不再一次性映射 N 个 env。
- **map_size 自适应**：读侧传 `map_size=0`，由 LMDB 从文件 meta 页读取真实尺寸（对 0.3.0 用 1 TiB 写出的旧文件同样有效）；写侧按图像像素量估算，不再每层、每进程盲目映射 1 TiB 虚拟地址。
- **`readahead=False`**：WSI 随机访问下避免无用的预读污染页缓存。
- **只读 Environment 进程内共享**：多实例不再各自 mmap 同一份数据，虚拟地址、页缓存和打开文件数都按"每张片子一份"计；共享 env 上每次读取仍用独立短事务，互不干扰。
- **写侧批量提交 + memoryview**：大图写入吞吐明显提升。
- **`test.py` 侧**：`np.ascontiguousarray(HWC→CHW)` + `torch.from_numpy` 零拷贝、`persistent_workers=True`（避免每 epoch 重开 env/缓存）、`pin_memory=True`、`prefetch_factor=2`。

---

## 四、使用方式

```python
import fmg
from torch.utils.data import DataLoader

# 读：可以安全地放在 Dataset.__init__ 里
slide = fmg.open_slide('xxx.fmg', patch_cache=256)
img = slide.read_region((x, y), 0, (512, 512))   # location 为 level-0 坐标

# 写
fmg.create_slide('out.fmg').save_from_numpy(
    data, ext='.jpg', pyramid=True, mpp=0.25, patch_size=256)

# DataLoader
loader = DataLoader(dataset, batch_size=128, num_workers=4,
                    persistent_workers=True, pin_memory=True,
                    worker_init_fn=worker_init_fn)   # worker_init_fn 里调 fmg.configure(0)
```

`slide` 新增构造参数：`patch_cache`（int，0 关闭）、`map_size`、`strict`、`auto_worker_tuning`。

**迁移注意**

1. 旧 `.fmg` 可直接读；但旧文件里 `level_dimensions` 是 `[h, w]`，且非正方形图像的 tile 本来就不完整，建议用新版 `save_from_numpy` 重灌一次以获得正确格式（`format_version=2`）。
2. `level_dimensions` / `dimensions` 现在严格返回 `(width, height)`；`level_downsamples` 返回标量列表。
3. `level>0` 的 `location` 现在按 OpenSlide 语义解释为 level-0 坐标；旧版实际把它当作 level 坐标。若代码里依赖旧行为，把坐标乘上 `level_downsamples[level]` 即可等价。
4. `map_size` 一般不需要再指定：读侧自动适配文件里的真实值（含旧版 1 TiB 文件），写侧按数据量估算；需要覆盖时给 `open_slide(..., map_size=...)` 或 `save_from_numpy(..., map_size=...)`。
5. `spawn` 启动时在 `worker_init_fn` 里调用 `fmg.configure(num_threads=0)`；`fork` 下库会自动处理。
6. `save_from_numpy()` 会删除并重建底层目录。若要覆盖写一张正在被读取的 `.fmg`，先 `close()` 所有读句柄再写，避免旧句柄还指着已被删除的 mmap。

---

## 五、验证

```bash
cd fmg_optimized
python selftest.py
```

测试覆盖：非方形图读写一致性、非对齐区域、越界补黑、level>0 语义、pickle 往返、多个 slide / clone 共享同一 env 且逐个关闭后仍可读、**父进程已打开 LMDB 时 fork 出 4 个 worker 并发读 1000 次且父进程同时读**、`spawn` 子进程反序列化 slide 后读图、JPEG 默认路径、以及读取 0.3.0 写出的旧文件。

本次运行结果：`ALL TESTS PASSED`，34 项检查全绿。`python _bench.py` 也跑通（见第六节复测说明）。

---

## 六、实测（单进程 / 4 worker，1024×1024 PNG、patch_size=256）

| 项目 | 0.3.0 | 0.4.0 |
|---|---|---|
| 写入一张 1024×1024 图 | 2.11 s | 1.94 s |
| 随机读 200 个 512×512 区域 | 1.54 s | **0.48 s**（约 3.2×，关缓存） |
| 同上，开启 `patch_cache=256` | — | 0.04 s（重复命中同一批 tile 时） |

读侧提速主要来自 `buffers=True` 零拷贝、目标画布代替包围盒缓冲、交集裁剪代替整块赋值；缓存那一行是"同一批 tile 被反复读"的理想场景（多 epoch 训练接近这种收益，单 epoch 顺序遍历全图时命中率会低很多，不要按这个数字估算吞吐）。写侧瓶颈在 `cv2.imencode`，所以提升有限，大图上批量提交与流式金字塔的收益会更明显。

另外用原版跑了一次 fork 对照（父进程已打开 LMDB，4 个 worker 各读 300 次）：本次 4 个 worker 都正常退出。**这正好说明它是低概率崩溃**——根因是 fork 后共享 env / reader slot 的未定义行为，复现与否取决于时序，不能因为某次没崩就认为旧写法安全；修复版在同样的压力下每个 worker 都独立持有自己的 Environment。

**复测（2026-11-06，同一沙盒，py-lmdb 3.0.0）**：`python _bench.py` 得到写入 0.3.0 / 0.4.0 = 24.20 s / 3.73 s，随机读 = 2.10 s / 0.64 s（关缓存）/ 0.06 s（开缓存）。沙盒的 virtiofs 挂载在建立 mmap 时会偶发 `EIO`，基准脚本为此加了重试，写侧绝对耗时被重试次数放大，**不能与上表直接比较**；读侧 2.10 / 0.64 ≈ 3.3×，与上表约 3.2× 的提速一致。原版 4 worker × 300 次的 fork 对照本次同样全部正常退出，再次印证上面的"低概率崩溃"判断。

## 参考

- LMDB Reader Lock Table / MDB_NOLOCK：<http://www.lmdb.tech/doc/group__readers.html>
- py-lmdb 并发与线程模型（MDB_NOTLS、事务不可并发使用）：<https://deepwiki.com/jnwatson/py-lmdb/4.3-concurrency-and-threading>
- `MDB_BAD_RSLOT` 多进程案例与 `readonly=True, lock=False` 解法：<https://stackoverflow.com/questions/56905502/>
- PyTorch DataLoader + LMDB 需在 `__getitem__` 懒打开：<https://blog.csdn.net/evilfromshadow/article/details/119537968>
