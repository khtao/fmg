# fmg

[![PyPI version](https://img.shields.io/pypi/v/fmg.svg)](https://pypi.org/project/fmg/)
[![Python versions](https://img.shields.io/pypi/pyversions/fmg.svg)](https://pypi.org/project/fmg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 把超大图像切成 tile 存进 LMDB，用 OpenSlide 的方式读，然后——在 `num_workers=8` 的 DataLoader 里，忘掉它的存在。

**fmg 0.4.0** · Python ≥ 3.7 · MIT · LMDB / OpenCV / NumPy / Pillow

面向数字病理全切片（WSI）、遥感大图、高分辨率扫描件这类「一张图几十 GB」的场景。它不把整图塞进内存，而是按 tile 随机读取，并提供一个你和 OpenSlide 一样熟悉的接口。

---

## 你有没有被这几个报错折磨过

```
DataLoader worker (pid 12345) exited unexpectedly
lmdb.BadRslotError: MDB_BAD_RSLOT
lmdb.Error: The environment '.../level0' is already open in this process
```

然后是更隐蔽的那种：**不报错，但非正方形图像写出来就是错的**——宽方向多出空 patch、高方向少了 tile，训练安静地喂进一堆黑块。

这些不是你的代码写错了。0.3.0 在父进程就打开 LMDB 并长期持有一个事务，`fork` 出来的 worker 共享同一个 env、同一段 mmap 和同一个 reader slot；再加一个 `[h, w]` 当成 `[w, h]` 的笔误，问题就齐了。

**0.4.0 把这一整类问题从库里拿掉了。** 你要做的只是把 `open_slide()` 放回 `Dataset.__init__`。

---

## 为什么值得换过来

### 🛡 多 worker 是安全的，不再靠运气

| 做法 | 效果 |
|---|---|
| 懒打开 | `open_slide()` 只读 JSON 元数据，不碰 LMDB；第一次 `read_region()` 才建环境 |
| 进程感知重开 | 检测到 pid 变化就丢弃继承来的句柄（**不调用 `close()`**，以免清掉父进程的 reader slot），在子进程重新 `lmdb.open()` |
| 短事务 | 每次读取一个独立只读事务，`max_spare_txns=8` 让创建近乎零成本，天然线程安全 |
| `lock=False` | MDB_NOLOCK 完全绕开 reader lock table，多进程只读没有 slot 竞争 |
| OpenCV fork 处理 | 子进程自动 `setNumThreads(0)` + 关闭 OpenCL，线程池不再在 `imdecode` 里挂住 |
| 可 pickle | 剔除 env 与锁，`fork` / `spawn` 两种启动方式都能用 |

### 🤝 同一张片子开几次，只占一份映射

进程内按 `(路径, map_size)` 共享**一个**只读 Environment，引用计数管理生命周期：`close()` 只减引用，最后一个使用者离开才真正关闭。于是这些场景全部不再冲突——

- `pickle` 往返产生的克隆对象
- train / val 两个 Dataset 指向同一张 slide
- 多个 Dataset 实例、多个 loader
- `spawn` 反序列化出的 slide

`os.register_at_fork` 会在子进程清空注册表并重建锁，避免 fork 时恰好有人持锁导致死锁。

### ⚡ 读得更快，而且省内存

零拷贝（`buffers=True` + `memoryview`）、直接分配目标画布 + 只拷贝相交 tile 的源/目标交集（不再先铺整个包围盒）、按 level 懒开环境、`map_size=0` 从文件 meta 页自动读真实尺寸、`readahead=False` 避免随机读时的无用预读。

写侧则是**流式金字塔**：每层由 level-0 一次 `cv2.INTER_AREA` 降采样得到、用完即弃，不再同时持有所有层的完整 numpy 副本（老写法峰值约 1.33× 原图，WSI 上就是几十 GB）；配合每 2048 个 tile 批量提交、`memoryview` 代替 `tobytes()`。

### ✅ 正确性与健壮性

- **宽高顺序彻底修正**：新格式统一 `(width, height)`，网格 x 遍历宽、y 遍历高；`h, w, _ = data.shape`；`level_downsamples` 存标量。
- **OpenSlide 语义**：`location` 是 level-0 坐标，`size` 是 level 坐标尺寸，内部按 downsample 缩放；越界区域补黑而不是切片错乱。
- **坏 tile 不拖垮训练**：缺 tile / 解码失败 / 形状不符默认涂黑并只警告一次；`strict=True` 改为抛 `CorruptTileError`。
- **不会留下半个文件**：元数据先写 `.tmp` 再 `os.replace()` 原子替换；`imencode` / `imdecode` 返回值全部检查。
- **向后兼容**：0.3.0 写出的旧 `.fmg` 直接可读，自动把对外视图翻回 `(w, h)`、把 downsample 取成标量并给一次警告；新版写入 `format_version=2`。
- `close()` 幂等、`__del__` 补齐、`with slide:` 正常可用；`mpp=None` 不再 `TypeError`；`os.path.splitext()` 代替硬切 `[:-4]`。

---

## 性能

1024×1024 图像、`patch_size=256`，随机读 200 个 512×512 区域：

| 项目 | 0.3.0 | 0.4.0 |
|---|---|---|
| 写入一张图 | 2.11 s | 1.94 s |
| 随机读 200 个区域 | 1.54 s | **0.48 s**（约 3.2×，关缓存） |
| 同上，`patch_cache=256` | — | 0.04 s |

复测（py-lmdb 3.0.0）读侧为 2.10 s → 0.64 s / 0.06 s，约 3.3×，与上表一致。

> **口径提醒**：带缓存那一行是「同一批 tile 被反复读」的理想场景，多 epoch 训练接近这个收益；单 epoch 顺序遍历全图时命中率会低很多，别按这个数字估吞吐。写侧瓶颈在 `cv2.imencode`，小图提升有限，大图上批量提交与流式金字塔的收益才明显。数字受硬件与文件系统影响，请以自己数据上的 `selftest.py` / `_bench.py` 为准。

另外，用原版跑 fork 对照时**有时 4 个 worker 会全部正常退出**——这恰恰说明它是低概率崩溃，取决于时序。别因为「我这儿没崩」就认为旧写法安全。

---

## 快速开始

### 安装

```bash
pip install fmg==0.4.0
```

从源码安装（或想跑自测 / 基准）：

```bash
git clone https://github.com/khtao/fmg.git && cd fmg
pip install .
python selftest.py
```

依赖会自动装上：`lmdb>=1.0`、`opencv-python>=4.0`、`numpy>=1.17`、`pillow>=7.0`。无 GUI 的服务器若装不上 `opencv-python`（缺 `libGL`），可在自己的环境里改用 `opencv-python-headless`。

想给项目做一次发布？作者操作手册见 [`RELEASE.md`](https://github.com/khtao/fmg/blob/main/RELEASE.md)。

### 写入一张图

```python
import numpy as np
import fmg

img = np.imread('huge.png')            # HWC，uint8 / 其它 dtype 均可

meta = fmg.create_slide('slide.fmg').save_from_numpy(
    img,
    ext='.png',          # jpg / jpeg / png / webp / bmp / tiff / tif
    pyramid=True,        # 用 INTER_AREA 生成金字塔，不再是隔点采样的锯齿
    mpp=0.25,            # 可省略（None 也合法）
    patch_size=256,
)
print(meta['level_dimensions'])        # [[w, h], [w/2, h/2], ...]
```

也可以只补写若干 tile：`save_from_tiles(patches, locations, level, ext, params=None, txn=None)`，外部传入 `txn=` 就能把成千上万个 tile 并进一次提交。

### 读取

```python
import fmg

slide = fmg.open_slide('slide.fmg', patch_cache=256)
w, h = slide.dimensions                       # (宽, 高)
print(slide.level_count, slide.level_dimensions, slide.level_downsamples)

patch = slide.read_region((1000, 2000), 0, (512, 512))   # -> PIL.Image
# location 是 level-0 坐标；size 是 level 坐标尺寸
low   = slide.read_region((1000, 2000), 1, (256, 256))   # 自动按 downsample 缩放

with slide:                                   # 或者 slide.close()
    pass
```

### 接进 PyTorch DataLoader

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
import fmg

class SlideDataset(Dataset):
    def __init__(self, path, size=512):
        # 放心放在 __init__：这里只读元数据，不打开 LMDB
        self.slide = fmg.open_slide(path, patch_cache=256)
        self.w, self.h = self.slide.dimensions
        self.size = size

    def __len__(self):
        return (self.w // self.size) * (self.h // self.size)

    def __getitem__(self, i):
        x = (i % (self.w // self.size)) * self.size
        y = (i // (self.w // self.size)) * self.size
        arr = np.asarray(self.slide.read_region((x, y), 0, (self.size, self.size)))
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

def worker_init_fn(worker_id):
    # fork 下库会自动处理；spawn 下必须显式调用这一句
    fmg.configure(num_threads=0)
    info = get_worker_info()
    if info is not None:
        seed = info.seed if info.seed is not None else torch.initial_seed()
        np.random.seed(seed % (2 ** 32))
        info.dataset.slide.read_region((0, 0), 0, (1, 1))   # 预热，错开首次开 env

loader = DataLoader(
    SlideDataset('slide.fmg'),
    batch_size=128, shuffle=True,
    num_workers=4,
    persistent_workers=True,     # 每 epoch 不重开 env / 不清缓存
    pin_memory=True,
    prefetch_factor=2,
    worker_init_fn=worker_init_fn,
    multiprocessing_context='fork',   # 换成 'spawn' 也行，slide 可 pickle
)
for batch in loader:
    ...
```

完整可运行版本见 [`test.py`](https://github.com/khtao/fmg/blob/main/test.py)。

---

## API 速查

| 入口 | 说明 |
|---|---|
| `fmg.open_slide(path, **kw)` | 打开已有 `.fmg`（只读，懒打开） |
| `fmg.create_slide(path, **kw)` | 创建写对象 |
| `fmg.configure(num_threads=0, use_opencl=False)` | 在 `worker_init_fn` 里显式配置 OpenCV |
| `slide.read_region(location, level, size)` | 返回 `PIL.Image`；越界补黑 |
| `slide.dimensions` / `level_dimensions` | `(w, h)` / `[[w, h], ...]` |
| `slide.level_count` / `level_downsamples` / `properties` / `patch_size` | 元数据 |
| `save_from_numpy(...)` / `save_from_tiles(...)` | 整图写入 / 追加 tile |
| `slide.close()` / `with slide:` | 幂等关闭，只释放本进程持有的引用 |
| `fmg.FmgError` / `fmg.CorruptTileError` | 异常类型 |
| `fmg.FORMAT_VERSION` | 当前为 `2` |

**`open_slide` 参数**

| 参数 | 默认 | 含义 |
|---|---|---|
| `patch_cache` | `128` | 进程内解码 tile 的 LRU 缓存大小，`0` 关闭 |
| `map_size` | `None` | 显式 LMDB map size；默认从文件 meta 页读真实值（含旧版 1 TiB 文件） |
| `strict` | `False` | 坏 tile 改为抛 `CorruptTileError` 而不是涂黑 |
| `auto_worker_tuning` | `True` | fork 子进程自动关闭 OpenCV 线程池 / OpenCL |

---

## 磁盘格式

```
slide.fmg                 JSON 元数据（格式版本、各层尺寸、downsample、mpp、编码参数…）
slide/
├── level0/               LMDB，key = str((x, y, x_max, y_max))
│   ├── data.mdb
│   └── lock.mdb
├── level1/
└── ...
```

每个 tile 由 `cv2.imencode` 编码、读时 `cv2.imdecode` 还原，二者互逆，通道顺序保持一致。格式版本 2 与 1 的目录布局相同，因此**旧文件可以直接读**。

---

## 从 0.3.0 迁移

1. 旧 `.fmg` 可以直接读，无需转换。但旧文件的 `level_dimensions` 是 `[h, w]`，且非正方形图像的 tile 本来就不完整——**建议用新版 `save_from_numpy` 重灌一次**以获得正确格式。
2. `level_dimensions` / `dimensions` 现在严格返回 `(width, height)`；`level_downsamples` 返回标量列表。
3. `level>0` 时 `location` 按 OpenSlide 语义解释为 level-0 坐标。若你的代码依赖旧行为（把 location 当 level 坐标），乘上 `level_downsamples[level]` 即可等价。
4. `read_region` 的签名是 `read_region(location, level, size)`，不是五个位置参数。
5. `map_size` 一般不用再指定：读侧自适应、写侧按数据量估算。
6. `spawn` 启动时记得在 `worker_init_fn` 里调 `fmg.configure(num_threads=0)`；`fork` 下自动处理。
7. 覆盖写一张正在被读的 `.fmg` 时，先 `close()` 所有读句柄——`save_from_numpy` 会删除并重建底层目录。

完整的根因分析、修复清单与迁移细节见 [`OPTIMIZATION.md`](https://github.com/khtao/fmg/blob/main/OPTIMIZATION.md)。

---

## 常见问题

**`patch_cache` 开多大？** 多 epoch 训练很值，单 epoch 顺序遍历全图命中率低。它是进程内的，不跨进程共享内存；显存紧张时给 `0` 也不会错。

**为什么 `spawn` 还要手动 `configure`？** 新进程里库检测不到「pid 变了」这件事（对象是被反序列化出来的，不是继承来的），所以 OpenCV 线程池需要你显式关掉。`fork` 路径下这一步是自动的。

**一个进程里开两个 slide 会冲突吗？** 不会，共享同一个 env。但别在还有读句柄时覆盖写同一个文件。

**`strict` 什么时候用？** 数据入库校验阶段打开它，坏 tile 立刻暴露；训练阶段保持默认 `False`，一个坏 patch 不该中断整轮训练。

**坏 tile 会静默吗？** 不会，第一次会发 `RuntimeWarning` 并打印 level 与 tile 坐标，之后只涂黑不刷屏。

---

## 验证

```bash
python selftest.py     # 35 项检查：非方形图一致性、越界补黑、level>0 语义、
                       # pickle / spawn / fork 4 worker 并发、共享 env、
                       # JPEG 路径、0.3.0 旧文件兼容
python _bench.py       # 0.3.0 / 0.4.0 读写对照基准
```

`selftest.py` 里有一组专门复现原崩溃场景的用例：**父进程已打开 LMDB，fork 出 4 个 worker 各读 250 次，父进程同时读 200 次**，每个 worker 都在自己的 pid 下重新持有 Environment。

---

## 参考

- LMDB Reader Lock Table / MDB_NOLOCK：<http://www.lmdb.tech/doc/group__readers.html>
- py-lmdb 并发与线程模型：<https://deepwiki.com/jnwatson/py-lmdb/4.3-concurrency-and-threading>
- `MDB_BAD_RSLOT` 多进程案例：<https://stackoverflow.com/questions/56905502/>
- PyTorch DataLoader + LMDB 懒打开：<https://blog.csdn.net/evilfromshadow/article/details/119537968>

原项目：<https://github.com/khtao/fmg> · 作者 khtao · MIT
