# fmg

[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://pypi.org/project/fmg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **把 OpenSlide 留在转换脚本里，把 fmg 放进 DataLoader。**
>
> 先用 OpenSlide 把厂商 WSI 解码一次、转成 LMDB tile store；之后每个训练 epoch、每个 worker 都只读本地 tile，不再碰 SVS/NDPI/MRXS 的解码路径。接口还是你熟悉的 `read_region(location, level, size)`。

**面向病理 AI 训练**：多 worker 随机抽 patch、多 epoch 反复读同一张全切片、数据放在本地 SSD 或网络盘上。本文是一份 **OpenSlide ↔ fmg 逐项对比 + 迁移指南**，包含可复制的代码、坐标契约、已知限制和核验方法。

> ⚠️ **发布状态（请先看）**：PyPI 上的 `fmg` 当前是 **0.3.0（2023-06-19）**，本文描述的 **0.4.0 尚未发布**（上传时遇到 403，处理中）。请按下面「安装」一节从源码构建，并在安装后执行 `python -c "import fmg; print(fmg.__version__)"` 确认版本——若得到 `0.3.0`，本文的 API 对齐、坐标修复和并发改造都不适用。
>
> 作者发布操作见 [`RELEASE.md`](https://github.com/khtao/fmg/blob/main/RELEASE.md)。

---

## 目录

1. [一句话结论：该不该换](#一句话结论该不该换)
2. [安装（0.4.0 待发布，从源码构建）](#安装040-待发布从源码构建)
3. [架构差异：为什么训练阶段要换读取层](#架构差异为什么训练阶段要换读取层)
4. [全维度对比表](#全维度对比表)
5. [多 worker 并发：因果链与库内实现](#多-worker-并发因果链与库内实现)
6. [迁移指南（四步，含 before/after 代码）](#迁移指南四步含-beforeafter-代码)
7. [坐标与像素一致性契约](#坐标与像素一致性契约)
8. [破坏性变更清单](#破坏性变更清单)
9. [性能：能承诺什么，不能承诺什么](#性能能承诺什么不能承诺什么)
10. [已知限制 / 不适合迁移的情况](#已知限制--不适合迁移的情况)
11. [API 速查与磁盘格式](#api-速查与磁盘格式)
12. [验证与参考](#验证与参考)

---

## 一句话结论：该不该换

| 你的情况 | 建议 |
|---|---|
| 同一批 WSI 要跑多个实验 / 多个 epoch，`num_workers ≥ 4` | **换**：转换一次，训练期只读本地 tile |
| 训练集在 NFS / 对象存储挂载上，OpenSlide 随机读慢或抖动 | **换**：把热数据转成本地 LMDB 副本 |
| DataLoader 里曾在 `__init__` 打开 OpenSlide 句柄并跨 worker 复用 | **换**：fmg 的 reader 生命周期按进程管理 |
| 只想偶尔打开一张片子看看、做可视化或人工标注 | **不换**：OpenSlide 直读更省事 |
| 源文件天天更新，无法承担一次转换成本 | **不换** |
| 依赖 ICC 色彩管理、C 层预乘 alpha 语义，或要把结果写回 SVS/DICOM | **暂缓**：这些 fmg 不提供 |
| 生产环境只允许装 PyPI 正式发布版 | **等 0.4.0 发布** |

**fmg 不是新的厂商格式阅读器。** SVS / NDPI / MRXS / SCN / DICOM 等仍由 OpenSlide 识别和解码；fmg 负责的是**转换之后的训练读取路径**。这个分工是全文的前提。

**迁入收益成立需要同时满足三个条件**（缺一条就继续用 OpenSlide 直读，或让两者共存）：

1. 厂商文件已经可靠地批量转换为 `.fmg`，原始 WSI 仍保留归档；
2. 训练访问以**随机小 patch + 多 epoch 重复读**为主，足以摊薄一次性转换成本并让 tile 缓存命中；
3. 项目可以接受 OpenSlide 在**非整数层级逐像素复现、ICC、alpha、多通道**上的能力缺口（见「已知限制」）。

一句更重要的话：**101/101 核验通过是迁移的验收条件，不是「两者完全等价」的数学证明。** 任何新扫描仪、新 vendor、新转换参数、新 OpenCV / LMDB 版本进生产前，都应重跑 API 对照、坐标对照、色彩 QA、多 worker 压力和端到端模型回归——按**逐文件、逐格式**验收，而不是一次性全局开关。

---

## 安装（0.4.0 待发布，从源码构建）

```bash
git clone https://github.com/khtao/fmg.git && cd fmg
pip install .
python -c "import fmg; print(fmg.__version__)"        # 必须输出 0.4.0
python selftest.py                                     # 期望 ALL TESTS PASSED
```

依赖会自动装上：`lmdb>=1.0`、`opencv-python>=4.0`、`numpy>=1.17`、`pillow>=7.0`。

需要从 OpenSlide 转换时，再装可选的转换与核验依赖：

```bash
pip install ".[openslide]"          # openslide-python + openslide-bin + tifffile
```

`openslide-bin` 带预编译的 OpenSlide C 库，省去系统包管理器；也可以只用 `conda install -c conda-forge openslide`。传一个**已打开的 slide 对象**给 `from_openslide()` 时不需要这些可选依赖。

### 无 GUI 的训练服务器 / 容器

OpenCV 官方要求**每个环境只安装 `opencv-python` 与 `opencv-python-headless` 中的一个**，共存会引发 `libGL.so.1` 缺失或行为异常。推荐在无 GUI 环境这样装：

```bash
pip install --no-deps .
pip install lmdb numpy pillow opencv-python-headless
```

或在 Dockerfile 里先 `pip install .` 再 `pip uninstall -y opencv-python && pip install opencv-python-headless`。注意 `setup.py` 声明的是 `opencv-python`，直接 `pip install .` 会拉 GUI 版。

### 版本与许可证

- fmg 自身：**MIT**。传递依赖的许可证需要项目自行核对（LMDB 为 OpenLDAP Public License，另有 OpenCV / NumPy / Pillow），MIT 不自动覆盖它们。
- OpenSlide 及其官方语言绑定：**LGPL-2.1（"not any later version"）**。这不等于"不可商用"，但修改库、替换动态库、容器分发等场景有对应源码与再链接义务，请由法务确认。
- 容器入口建议打印 `openslide.__version__`、`openslide.__library_version__`、`fmg.__version__`；只锁 pip 包版本无法排除系统 `libopenslide` 漂移。

---

## 架构差异：为什么训练阶段要换读取层

```
传统：  SVS ──OpenSlide 解码──▶ worker1 / worker2 / ... / workerN   （每个 patch 都走厂商解码路径）
fmg：   SVS ──OpenSlide 解码──▶ [一次性转换] ──▶ LMDB tiles
                                          └──▶ worker1 / ... / workerN   （只读本地 tile + OpenCV 解码）
```

PyTorch 官方文档提醒：`num_workers>0` 时 worker 会复制父进程中可访问的 Python 对象，Dataset 过大时总内存近似 `num_workers × 父进程占用`。把一个持有 C 库 mmap、文件描述符、GLib 缓存或解码状态的 OpenSlide 句柄放在 `Dataset.__init__`，再让 DataLoader 复制给 worker，得到的是「继承来的、未重新初始化的状态」，而不是干净对象。

**措辞需要精确**：OpenSlide 官方文档没有给出「句柄可安全跨 fork」的保证，也没有发布「fork 后必然崩溃」的结论。因此本文把它定性为**高风险反模式（状态继承机制 + 社区经验）**，而不是官方缺陷。稳妥写法是「在 worker 内打开」，PyTorch 也建议用 `spawn` / `forkserver` 获得干净初始化。

fmg 的做法是把状态管理换成 LMDB 环境的进程生命周期管理：py-lmdb 官方明确规定 **Environment 不能跨 `fork()` 使用**，子进程的 mmap、lock table 与 fd 无效，必须在子进程重新打开；继承的只读事务还会占用 reader lock-table slot。fmg 的实现正是围绕这几条约束设计的（见下一节）。

---

## 全维度对比表

> 「fmg」列描述的是本仓库 0.4.0 源码 + 本地实测；「OpenSlide」列以官方 API / 格式 / 属性文档与 C 源码为依据。版本口径：OpenSlide C 4.x / openslide-python 1.4.x。

### 1. 定位与工作流

| 维度 | OpenSlide | fmg 0.4.0 | 迁移含义 |
|---|---|---|---|
| 角色 | 厂商 WSI **只读解码器** | 转换后的 **LMDB tile store** | 两者是前后关系，不是替代关系 |
| 直接读 SVS/NDPI/MRXS/SCN/DICOM 等 | 支持（官方格式页列出十余种，且声明"may be incomplete"） | **不支持**，必须先转换 | 转换前用 `OpenSlide.detect_format()` 逐文件审计 |
| 预处理成本 | 无 | **有**：OpenSlide 解码 + tile 编码 + LMDB 写入，磁盘多一份副本 | 不是零成本；原始 WSI 必须保留归档 |
| 训练期格式探测 | 每次打开都要解析厂商文件 | 只读一份 JSON 元数据 | 多 epoch / 多实验时摊薄成本 |
| 可写回厂商格式 | 官方 Python API 是 read-only | 不能生成合规 SVS/DICOM/MRXS | 标注输出请用 TIFF / Zarr / COCO / GeoJSON |
| 缓存粒度 | OpenSlide 4 的 tile cache 按**字节**计容量，可创建/共享/释放 | 进程内 tile LRU 按 **tile 个数**计（`patch_cache`） | `set_cache()` 在 fmg 是 no-op，别照搬容量数字 |

### 2. API 逐项（这是"改 import 就能跑"的底气）

| OpenSlide | fmg 0.4.0 | 差异 / 迁移动作 |
|---|---|---|
| `openslide.open_slide(path)` | `fmg.open_slide(path)` | 换成 `.fmg` 路径；懒打开、fork 安全 |
| `dimensions` | 同，`(w, h)` tuple | 无改动 |
| `level_count` | 同 | 无改动 |
| `level_dimensions` | 同，**tuple of `(w,h)` tuple** | 0.4.0 早期返回 list；序列化/断言代码请检查 |
| `level_downsamples` | 同，**tuple of float** | 非整数 downsample 原样保留；用 `math.isclose` 比较 |
| `properties` | 同，Mapping | 含 `openslide.*` 键；转换时**透传源片子全部属性**。fmg 私有键（`mpp` / `patch_size` / `level_dimensions`）保留原生 JSON 类型，不是字符串 |
| `associated_images` | 同，`name → RGBA Image` 惰性 Mapping | label / macro / thumbnail 转存为 PNG；未知名称一并保留 |
| `color_profile` | `ImageCmsProfile` 或 `None` | **fmg 恒为 None**（不存 ICC）← 明确差异 |
| `read_region(loc, level, size)` | 同签名、同坐标系 | 默认返回 **RGB**；`rgba=True` 返回 RGBA。见「坐标与像素一致性契约」 |
| `get_best_level_for_downsample(d)` | 同 | 直接照搬 C 源码 `openslide_get_best_level_for_downsample` 的循环；边界 `d<1` 与超大 `d` 已覆盖 |
| `get_thumbnail(size)` | 同 | 同样的背景色 + LANCZOS 路径 |
| `set_cache(cache)` | 接受 `OpenSlideCache` | fmg 为 **no-op**（自带 LRU），容量单位与共享范围不同 |
| `close()` / `with slide:` / 析构 | 同 | fmg 幂等，只释放本进程引用 |
| `OpenSlide.detect_format(path)` | `fmg.detect_format(path)` + 类方法 | **语义不同**：OpenSlide 返回 vendor 字符串，fmg 返回 `'fmg'` / `None` |
| `OpenSlideError` / `OpenSlideUnsupportedFormatError` | `fmg.OpenSlideError` / `fmg.OpenSlideUnsupportedFormatError` | openslide 已安装时**直接复用其异常类**，`except openslide.OpenSlideError` 能捕获 fmg 的错 |
| latching error（一次出错后除 `close()` 外持续失败） | 官方明确语义 | **fmg 不 latching**：坏 tile 默认涂黑 + 只警告一次，不中断整轮训练 |
| `PROPERTY_NAME_*` 常量（11 个） | `fmg.PROPERTY_NAME_*` 同名 | 无改动 |

### 3. 多 worker 与进程模型（病理训练最关心的一栏）

| 维度 | OpenSlide | fmg 0.4.0 |
|---|---|---|
| 官方跨 fork 保证 | **未给出** | 不适用：LMDB 官方明确禁止跨 fork 继承 Environment |
| 常见风险写法 | 主进程 `__init__` 打开句柄 → worker 继承 | 0.3.0 曾复现 `MDB_BAD_RSLOT` / "already open in this process"；0.4.0 已修 |
| 句柄生命周期 | 由调用方管理 | 懒打开 + 按 pid 感知重开 + 短事务；`os.register_at_fork` 清理注册表 |
| 同一路径开多个对象 | 各对象独立句柄 | 进程内共享**一个**只读 Environment，引用计数；`close()` 只减引用 |
| `spawn` / pickle | 需自行处理 | slide 可 pickle，fork / spawn 都可用；spawn 下在 `worker_init_fn` 调 `fmg.configure(num_threads=0)` |
| 并发读取 | 依赖格式驱动与私有缓存 | MDB_NOLOCK + 每次读取独立只读事务 + `max_spare_txns=8`；读者不阻塞读者 |
| OpenCV 线程池 | 不涉及 | fork 子进程自动 `setNumThreads(0)` + 关闭 OpenCL |

> 注意：MDB_NOLOCK 意味着**调用方自己管理锁**。fmg 的适用前提是「转换完成后只读、不再写入同一目录」；不要在有读句柄时覆盖写同一个 `.fmg`。

### 4. 坐标、像素与通道

| 维度 | OpenSlide | fmg 0.4.0 |
|---|---|---|
| `location` 坐标系 | **level 0**（官方明确） | 同 |
| `size` 坐标系 | **请求 level**（官方明确） | 同 |
| level-0 → level 换算 | C 源码 `int64_t = x / downsample`，向零截断；非整数 ds + cairo 裁剪会产生约 1px 抖动 | `int(x0 / ds)`，与 C 一致；0.4.0 早期用 `round()` 会错开 1 像素，已修 |
| 负坐标 / 越界 | C 层把负坐标转成画布偏移；Python 未承诺填充颜色 | 补黑；`rgba=True` 时越界为 `(0,0,0,0)` |
| 返回图像 | **RGBA**，Python 层非预乘；C 层是预乘 ARGB | **RGB**（默认）；`rgba=True` 给 RGBA |
| 透明空白区域 | `(0,0,0,0)`，直接裁 alpha 会变黑而非想要的白色；背景色取 `openslide.background-color` | 默认转换丢弃 alpha → 黑色；需要白背景请在转换前合成 |
| 非整数 downsample | 常见 4.0001 / 16.0005，不能假定 `2**level` | 原样保留，不要硬编码层级倍率 |

### 5. 元数据、依赖、许可证

| 维度 | OpenSlide | fmg 0.4.0 |
|---|---|---|
| `openslide.mpp-x/y` | 官方注明"may not be present or accurate" | 透传；同时解析进 fmg 的 `mpp` 字段 |
| `objective-power` / `quickhash-1` | 官方注明前者常不准确、后者**不是完整性校验** | 原样透传，不做语义增强 |
| ICC profile | OpenSlide 4 起支持 | **不保留**（`color_profile is None`） |
| 厂商私有属性（`aperio.*` / `dicom.*`） | 提供 | 全部透传到独立 `openslide_properties` 字典 |
| 系统依赖 | C 库 `libopenslide`（可用 `openslide-bin` / conda） | 纯 Python + `lmdb` / OpenCV / NumPy / Pillow |
| 许可证 | LGPL-2.1（not any later version） | MIT（自身）；传递依赖另计 |
| 服务器部署 | 需要 C 库 | 推荐 `opencv-python-headless`，每环境只装一个 OpenCV 包 |

---

## 多 worker 并发：因果链与库内实现

**风险链（OpenSlide 侧，工程推断 + 官方机制）**

```
Dataset.__init__ 打开 OpenSlide 句柄
  → DataLoader fork/spawn 复制 Dataset
  → worker 拿到继承的 C 句柄 / mmap / 缓存 / 锁状态
  → 状态未重新初始化 → 行为依赖具体格式驱动、版本与文件系统
  → 官方未给跨 fork 保证 → 属于高风险反模式
```

**fmg 侧的实现（本仓库源码可证）**

| 机制 | 作用 |
|---|---|
| 懒打开 | `open_slide()` 只读 JSON 元数据，不碰 LMDB；第一次 `read_region()` 才建环境 |
| 进程感知重开 | 检测 pid 变化时**丢弃**继承句柄（不调用 `close()`，避免清掉父进程的 reader slot），在子进程重新 `lmdb.open()` |
| 注册表 + 引用计数 | 同进程按 `(realpath, map_size)` 共享一个只读 Environment；train/val 双 Dataset、pickle 克隆、多个 loader 不再冲突 |
| `register_at_fork(after_in_child=...)` | 子进程清空注册表、重建锁、只丢弃继承句柄而不 close，防止 fork 时持锁导致死锁 |
| 短事务 | 每次读取一个独立只读事务；不会长期占用 reader lock-table slot |
| `lock=False` (MDB_NOLOCK) | 只读场景绕开 reader lock table；由调用方保证"转换后不写" |
| OpenCV fork 处理 | 子进程自动 `setNumThreads(0)` + 关闭 OpenCL |
| 可 pickle | `__getstate__` 剔除 env 与锁，fork / spawn 均可 |

`selftest.py` 里有一组专门复现旧崩溃的用例：**父进程已打开 LMDB，fork 出 4 个 worker 各读 250 次，父进程同时读 200 次**，每个 worker 都在自己的 pid 下重新持有 Environment；另有一组 `spawn` 反序列化用例。当前 `ALL TESTS PASSED`。

---

## 迁移指南（四步，含 before/after 代码）

### Step 0 · 审计源片子（别跳过）

OpenSlide 官方声明其格式支持"may be incomplete"，先确认每张片子能被识别、有哪些层级、缺哪些属性：

```python
import openslide

path = 'slide.svs'
vendor = openslide.OpenSlide.detect_format(path)     # 返回 vendor 或 None
with openslide.open_slide(path) as osr:
    print('vendor   :', osr.properties.get('openslide.vendor'))
    print('levels   :', osr.level_dimensions, osr.level_downsamples)
    print('mpp      :', osr.properties.get('openslide.mpp-x'))
    print('bg       :', osr.properties.get('openslide.background-color'))
    print('associated:', list(osr.associated_images))
    print('ICC      :', osr.color_profile is not None)
```

把 `vendor / levels / mpp / 背景色 / 关联图 / ICC / 异常` 存成清单。转换失败、属性缺失、关联图缺失都应当是**可审计的失败**，不要静默降级成空 metadata。对临床数据，建议同时记录源 SHA-256、OpenSlide 版本、fmg 版本、转换参数与时间戳。

### Step 1 · 一行转换（默认无损 PNG）

```python
import fmg

meta = fmg.from_openslide('slide.svs', 'slide.fmg', patch_size=256)
print(meta['level_dimensions'], meta['level_downsamples'])
```

保留：所有 level 的精确尺寸与 downsample、源 `properties`（含 `openslide.mpp-x` 等）、`associated_images`（label / macro / thumbnail）、`mpp`。默认 `ext='.png'` 无损；改 `.jpg` 会有压缩损失，不要声称像素无损。

也可以传已打开的对象（此时不需要装 openslide 包）：

```python
with openslide.open_slide('slide.svs') as osr:
    fmg.create_slide('slide.fmg').save_from_openslide(
        osr, ext='.png', patch_size=256,
        include_associated=True,
        progress=lambda lv, done, total: None,   # 可选进度回调
    )
```

**转换成本的实话**：这一步要跑 OpenSlide 解码 + tile 编码 + LMDB 写入，会占用 CPU 与额外磁盘；`max_level_bytes=256<<20` 以下的层一次读整层，超大层按 2048 分块读。**原始 WSI 请保留归档**，fmg 副本是训练缓存，不是替代品。

### Step 2 · 改读取代码（before / after）

```python
# ---------- before: OpenSlide ----------
import openslide
slide = openslide.open_slide('slide.svs')
patch = slide.read_region((1000, 2000), 0, (512, 512)).convert('RGB')
level = slide.get_best_level_for_downsample(4.0)
mpp   = slide.properties.get('openslide.mpp-x')

# ---------- after: fmg ----------
import fmg
slide = fmg.open_slide('slide.fmg', patch_cache=256)     # 路径换成 .fmg
patch = slide.read_region((1000, 2000), 0, (512, 512))   # 已是 RGB，无需 .convert()
level = slide.get_best_level_for_downsample(4.0)          # 不变
mpp   = slide.properties.get('openslide.mpp-x')           # 不变
```

需要保留 RGBA 时用 `fmg.open_slide(path, rgba=True)`，越界像素为 `(0,0,0,0)`，与 OpenSlide 的透明背景一致。

**关于透明区域 / 背景色**：fmg 默认存 RGB，会把 OpenSlide 的透明空白变成 `(0,0,0)` 黑。OpenSlide 官方文档明确警告，C 层预乘 alpha 的透明像素是 `(0,0,0,0)`，直接裁掉 alpha 得到的是黑色而不是常见的白色玻璃背景。若你的模型对背景敏感，转换前先按 `openslide.background-color` 合成：

```python
import openslide, fmg
from PIL import Image

class OpaqueSlide:
    """把 OpenSlide 的 RGBA 合成到不透明背景，再交给 fmg。"""
    def __init__(self, path):
        self._osr = openslide.open_slide(path)
        self.level_dimensions = self._osr.level_dimensions
        self.level_downsamples = self._osr.level_downsamples
        self.properties = self._osr.properties
        self.associated_images = self._osr.associated_images
        self._bg = '#' + self.properties.get(
            'openslide.background-color', 'ffffff').lstrip('#')

    def read_region(self, location, level, size):
        rgba = self._osr.read_region(location, level, size)
        out = Image.new('RGB', rgba.size, self._bg)
        out.paste(rgba.convert('RGB'), mask=rgba.getchannel('A'))
        return out

fmg.from_openslide(OpaqueSlide('slide.svs'), 'slide.fmg')
```

### Step 3 · 接进 DataLoader（可直接复制）

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
        cols = self.w // self.size
        x, y = (i % cols) * self.size, (i // cols) * self.size
        arr = np.asarray(self.slide.read_region((x, y), 0, (self.size, self.size)))
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

def worker_init_fn(worker_id):
    # fork 下库会自动处理；spawn 下必须显式配置 OpenCV
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
    multiprocessing_context='fork',   # 换 'spawn' 也行，slide 可 pickle
)
for batch in loader:
    ...
```

完整可运行版本见 [`test.py`](https://github.com/khtao/fmg/blob/main/test.py)。`spawn` 会重新导入主模块，Dataset 与路径必须可导入、不要在 `__main__` 里做初始化；子进程用 CUDA 时 PyTorch 官方也建议 `spawn` / `forkserver`。

### Step 4 · 跑回归核验

```bash
python selftest.py                 # 主回归，期望 ALL TESTS PASSED
python verify_openslide_compat.py  # 用真实 OpenSlide 核验 API / 坐标 / 像素，生成 compat_report.md
```

---

## 坐标与像素一致性契约

**契约（与 OpenSlide 官方 API 一致）**：`location` 是 level-0 左上角，`size` 是请求 level 的 `(width, height)`；换算 `lx = int(x0 / downsample)`（向零截断，与 OpenSlide C 源码一致）。

**可复现坐标写法**（任意 downsample 下都拿到确定的 level 坐标 `lx`）：

```python
import math
X0 = math.ceil(lx * downsample)   # 保证 int(X0 / ds) == lx
region = slide.read_region((X0, Y0), level, (w, h))
```

整数 downsample 直接 `X0 = lx * ds`。**不要硬编码 `level_downsamples[l] == 2**l`**，Aperio / 通用 TIFF 上常见 4.0001、16.0005 这类实际值。

### 本仓库实测（不是通用保证）

`verify_openslide_compat.py` 用真实 OpenSlide（Python 1.4.7 / C 库 4.0.1）读两张自建金字塔 tiled TIFF，再与转换后的 fmg 逐像素比较；生成 `compat_report.md`。环境：Python 3.10、py-lmdb 3.0.0、OpenCV 5.0.0。

| 检查 | 整数 ds（1 / 2 / 4） | 非整数 ds（1 / 1.5356 / 3.0735） |
|---|---|---|
| 每个 level 整层内容（丢弃 alpha 比 RGB） | 3/3 level 逐像素一致 | 3/3 level 逐像素一致 |
| level 0 任意坐标（随机 / 跨 tile / 负 / 越界 / 底边） | 全部一致 | 全部一致 |
| level>0 对齐坐标 core cases（强制断言） | **36/36 一致** | 22/36（列为 information） |
| level>0 任意 level-0 坐标 | 14/36 一致 | 4/36 一致 |
| `rgba=True`，level 0 含 alpha | 8/8 一致 | 8/8 一致 |
| 合成对象：properties / 关联图往返 | — | 54/54 全 level 一致，label/macro 像素级往返 |

总判定 `RESULT: PASS`，**101/101 强制检查通过**；`selftest.py` 另含 50 项检查全绿（含 OpenSlide 接口表面用例）。

**为什么 level>0 的任意坐标会有差异？** 这是 OpenSlide 自身在 level 坐标量化 / cairo 裁剪时的约 1px 取整行为：官方邮件列表早已记录「请求两个任意偏移的重叠子区域时，同一全局位置的像素值每通道可差至多 1」，`openslide#222` 建议始终用 level-0 坐标避免自行缩放引入额外取整，`tiffslide#51` 则实测了「统一 downsample 因子」与「分别按宽高比例」两条路径会给出 `(24,24)` vs `(25,24)`。不同厂商格式与版本表现不同，**任何第三方实现都无法用一条公式精确复现**。fmg 选择文档化的 C 标准语义 `int(x / ds)`，因为它对 level 0 恒等、对整数 downsample 的对齐坐标完全确定、对自身行为可预测可缓存。

**怎么解读差异**：如果训练以随机裁剪、颜色增强、归一化为常规操作，level>0 上少量像素位移通常不是主要误差源；若任务要求与 OpenSlide 产生完全相同的 tile 哈希、回归快照或诊断渲染，fmg 不能替代 OpenSlide。

**因此**：训练取 patch 优先用 level 0；需要严格一致的多尺度评估时用 `ceil(lx * ds)` 构造坐标，或先把源片子重采样成整数 downsample 的金字塔再转换；**不要拿 level>0 的任意 level-0 坐标做逐像素回归断言**。完整对齐表与归因分析见 [`OPENSLIDE_COMPAT.md`](https://github.com/khtao/fmg/blob/main/OPENSLIDE_COMPAT.md)。

**"无损"的边界**：默认 PNG 编码可保证"解码后 RGB 逐像素一致"，但透明区域合成、背景色、ICC、JPEG 源的非精确缩放与边界填充都可能改变渲染结果；换 `.jpg` 则连像素无损也不成立。跨扫描仪 batch 的可比性还涉及 ICC，fmg 不保留 profile。

---

## 破坏性变更清单

| # | 变化 | 老代码怎么办 |
|---|---|---|
| 1 | `level_dimensions` / `level_downsamples` 从 list 改为 **tuple** | 断言 `== [..]` 的地方改成 tuple；用 `math.isclose` 比 downsample |
| 2 | level>0 换算从 `round()` 改为 `int(x0/ds)` 截断 | region 最多移动 1 像素；用 `ceil(lx*ds)` 构造坐标 |
| 3 | `read_region` 默认 **RGB**，不再 RGBA | 删掉 `.convert('RGB')`；需要 alpha 用 `rgba=True` 或转换前合成背景 |
| 4 | 越界 / 非法 level / 零尺寸抛 **`OpenSlideError`** 而非 `ValueError` | `except` 改成 `fmg.OpenSlideError` |
| 5 | `color_profile` 恒为 `None` | 色彩管理在转换前完成，或自行保存源 ICC |
| 6 | `set_cache()` 是 no-op，缓存单位从字节变 tile 个数 | 用 `patch_cache=N` 参数 |
| 7 | `detect_format()` 返回 `'fmg'` / `None`，不是 vendor | 格式识别仍在转换阶段用 OpenSlide 做 |
| 8 | 不做 latching error | 坏 tile 默认涂黑 + 一次警告；入库校验用 `strict=True` |
| 9 | 源文件与 `.fmg` 是两份数据 | 保留原始 WSI 归档；覆盖写前 `close()` 所有读句柄 |
| 10 | 0.3.0 的旧 `.fmg` 可读但格式有缺陷 | 旧文件 `level_dimensions` 是 `[h,w]` 且非方形图可能缺 tile，建议重灌 |

---

## 性能：能承诺什么，不能承诺什么

**能承诺的是架构**：厂商解码从每个 worker 的读取路径移出、LMDB mmap + 短事务多读者、进程内 tile LRU 在多 epoch 重复抽样时避免重复解码。

**不能承诺的是倍数**：本次未找到 OpenSlide 或 fmg 官方针对 DataLoader、多 worker、不同 WSI 格式的端到端微基准。吞吐取决于 page cache、存储类型（本地 NVMe / SATA SSD / NFS）、patch 分布与复用率、worker 数、`pin_memory`、prefetch 与 OpenCV 解码，**不是库名**。请在自己的数据上做 A/B，报告 P50/P95/P99 延迟、峰值 RSS、worker 启动时间与失败率。

本仓库自带的 `_bench.py` 是 **fmg 0.3.0 → 0.4.0 的版本自比**，**不是与 OpenSlide 的对比**，请勿据此宣称"比 OpenSlide 快 N 倍"：

| 项目（1024×1024 合成图、patch 256、随机读 200 个 512×512 区域） | fmg 0.3.0 | fmg 0.4.0 |
|---|---|---|
| 写入一张图 | 2.11 s | 1.94 s |
| 随机读 200 个区域 | 1.54 s | 0.48 s（关缓存） |
| 同上，`patch_cache=256` | — | 0.04 s |

复测（py-lmdb 3.0.0）读侧 2.10 s → 0.64 s / 0.06 s，与上表同量级。**两套数字来自不同批次，不能拼成一条性能曲线，更不能把 0.3.0 与 0.4.0 的缓存数字混用**。带缓存那行是"同一批 tile 被反复读"的理想状态（衡量命中后的下限）：多 epoch 持续访问相同区域时接近它，但单 epoch 内按患者 / 组织 / 空间位置随机采样时 tile 可能只访问一次，命中率会显著下降。写侧瓶颈在 `cv2.imencode`；本次复测的写入绝对耗时还被沙盒 virtiofs 偶发 EIO 重试放大过一个数量级（24.20 s / 3.73 s），不能与文档原表直接比较——生产 benchmark 应同时记录存储类型、冷/热 page cache、IOPS、进程数、tile 尺寸与异常重试。

> 补充：用旧版跑 fork 对照时**有时 4 个 worker 全部正常退出**——这恰恰说明它是低概率、依赖时序的崩溃，别因为"我这儿没崩"就认为旧写法安全。

---

## 已知限制 / 不适合迁移的情况

- **0.4.0 尚未发布**：PyPI 当前仍是 0.3.0；本文所有能力以源码构建后的 `fmg.__version__ == '0.4.0'` 为前提。
- **不直接读厂商格式**：SVS / NDPI / MRXS / SCN / DICOM 等必须先经 OpenSlide 转换；OpenSlide 打不开的文件，fmg 也帮不了。
- **不保留 ICC profile**：`color_profile is None`；跨扫描仪色彩一致性需在转换前处理。
- **默认丢弃 alpha**：透明空白变黑；依赖白背景的数据要先合成（见 Step 2）。
- **非整数 downsample 的 level>0 有约 1px 抖动**：这是 OpenSlide 侧行为，fmg 不消除；规避办法见上一节。
- **转换非零成本、磁盘多一份**：需保留原始 WSI 归档，并记录可复现的转换清单。
- **不能写回原格式**：fmg 的 LMDB 不是厂商回写器。标注输出请用 TIFF / Zarr / COCO / GeoJSON。
- **不适合荧光 / 多通道 / 必须保留原始 alpha 语义的数据**：`.fmg` 固定 3 通道 RGB tile，关联图虽存 PNG，但透明语义应在转换前明确合成策略并写成测试。
- **MDB_NOLOCK 需自律**：只读场景可靠；不要在有读句柄时覆盖写同一目录，也不要让多个 writer 与它并存。
- **网络文件系统与热更新**：LMDB 官方不推荐在 NFS 上使用；「训练期边读边写、热更新 tile」路径未经压测，请把转换与训练在流程上分开。
- **`openslide.mpp-x` / `objective-power` / `quickhash-1` 不保证可靠**：官方注明 MPP 可能缺失或不准确、objective 常不准确、quickhash **不是完整性校验**。厂商私有属性因格式而异，不能承诺全部可恢复。
- **LMDB 不是"无限并发"**：reader table、lock file、文件系统行为、`map_size`（虚拟地址预留，部分文件系统会预分配）都有限制；报告基准时请同时给出 RSS 与磁盘占用。
- **能力缺口要留逃生通道**：色彩管理、透明区域、多通道、原始文件渲染与精确对齐应由 OpenSlide 路径处理；若生产只能用 fmg，请显式禁用这些场景，而不是静默降级。

---

## API 速查与磁盘格式

| 入口 | 说明 |
|---|---|
| `fmg.open_slide(path, **kw)` | 打开已有 `.fmg`（只读，懒打开） |
| `fmg.create_slide(path, **kw)` | 创建写对象 |
| `fmg.from_openslide(src, path, **kw)` | 把 OpenSlide 片子 / 任意 AbstractSlide-like 对象转成 `.fmg` |
| `fmg.detect_format(path)` | 合法 `.fmg` 返回 `'fmg'`，否则 `None` |
| `fmg.configure(num_threads=0, use_opencl=False)` | 在 `worker_init_fn` 里显式配置 OpenCV |
| `slide.read_region(location, level, size)` | `PIL.Image`；level-0 坐标；越界补黑 |
| `slide.dimensions` / `level_dimensions` / `level_downsamples` | tuple |
| `slide.properties` / `associated_images` | 含 `openslide.*` 键 / 惰性 RGBA 映射 |
| `slide.get_best_level_for_downsample(d)` / `get_thumbnail(size)` | OpenSlide 同语义 |
| `save_from_numpy(...)` / `save_from_openslide(...)` / `save_from_tiles(...)` | 整图 / 转换 / 追加 tile |
| `slide.close()` / `with slide:` | 幂等关闭 |
| `fmg.OpenSlideError` / `fmg.FmgError` / `fmg.CorruptTileError` | 异常 |
| `fmg.PROPERTY_NAME_*` / `fmg.FORMAT_VERSION` | 属性名常量 / 当前为 `2` |

**`open_slide` 参数**：`patch_cache=128`（tile 个数，0 关闭）、`map_size=None`（从文件 meta 页读真实值）、`strict=False`（坏 tile 涂黑并警告）、`auto_worker_tuning=True`（fork 子进程关 OpenCV 线程池）、`rgba=False`（返回 RGB）。

**`from_openslide` / `save_from_openslide` 参数**：`ext='.png'`、`params=None`、`patch_size=256`、`commit_every=2048`、`map_size=None`、`max_level_bytes=256<<20`、`include_associated=True`、`progress=None`。

**磁盘布局**（格式版本 2，与 0.3.0 的目录结构相同，旧文件仍可读）：

```
slide.fmg                 JSON 元数据（各层尺寸、downsample、mpp、编码参数、源属性…）
slide/
├── level0/               LMDB，key = str((x, y, x_max, y_max))
│   ├── data.mdb
│   └── lock.mdb
├── level1/
├── associated/           label.png / macro.png / ...
└── ...
```

tile 由 `cv2.imencode` 编码、`cv2.imdecode` 还原，二者互逆；元数据先写 `.tmp` 再 `os.replace()` 原子替换。

**常见问题**

- **`patch_cache` 开多大？** 多 epoch 训练值得开，单 epoch 顺序遍历命中率低。它是进程内 LRU，不跨进程共享内存。
- **为什么 `spawn` 要手动 `configure`？** 对象是反序列化出来的，库检测不到 pid 变化，OpenCV 线程池需显式关掉；fork 路径自动处理。
- **一个进程开两个 slide 会冲突吗？** 不会，共享同一个 env；但别在还有读句柄时覆盖写同一文件。
- **`strict` 什么时候用？** 入库校验阶段打开让坏 tile 立刻暴露；训练保持 `False`。
- **坏 tile 会静默吗？** 不会，首次发 `RuntimeWarning` 并打印 level 与 tile 坐标。

**从 0.3.0 迁移的完整根因分析**见 [`OPTIMIZATION.md`](https://github.com/khtao/fmg/blob/main/OPTIMIZATION.md)。

---

## 验证与参考

```bash
python selftest.py                 # 50 项：读写一致性、OpenSlide 接口表面、
                                   # fork/spawn 并发、共享 env、旧文件兼容
python verify_openslide_compat.py  # 真实 OpenSlide 核验 API/坐标/像素，生成 compat_report.md
python _bench.py                   # fmg 0.3.0 / 0.4.0 版本自比基准（非 vs OpenSlide）
```

`verify_openslide_compat.py` 需要 `pip install ".[openslide]"`；它自己生成测试金字塔 TIFF，不依赖外部数据。

**权威来源**

- OpenSlide Python API（属性、方法、latching、alpha、ICC）：<https://openslide.org/api/python/>
- OpenSlide C 源码 `openslide.c`（`read_region_area` / `get_best_level_for_downsample`）：<https://github.com/openslide/openslide/blob/main/src/openslide.c>
- OpenSlide C 头文件 `openslide.h`（「除 `openslide_close()` 外线程安全」、属性宏、tile cache）：<https://raw.githubusercontent.com/openslide/openslide/openslide-4.0.1/src/openslide.h>
- OpenSlide 支持的格式清单（含 "may be incomplete" 声明）：<https://openslide.org/formats/generic-tiff>
- 预乘 ARGB 与透明背景（「直接丢 alpha 会得到黑边」）：<https://openslide.org/docs/premultiplied-argb>
- 已知属性说明（MPP / objective / quickhash 的可靠性）：<https://openslide.org/docs/properties>
- OpenSlide 许可证 LGPL-2.1 及复合许可证声明：<https://openslide.org/> · <https://github.com/openslide/openslide-python/blob/main/pyproject.toml>
- OpenSlide 4.0 API 变化（新增 ICC profile 相关 API）：<https://github.com/openslide/openslide/releases/tag/v4.0.0>
- py-lmdb 并发、fork 与 reader slot：<https://pylmdb.readthedocs.io/en/release/>
- LMDB 官方：「Use an `MDB_env*` in the process which opened it, **not after fork()**」、长事务与 reader slot：<http://www.lmdb.tech/doc/>
- MDB_NOLOCK 说明：<http://www.lmdb.tech/doc/group__mdb__env.html>
- PyTorch DataLoader 多进程内存与 worker：<https://pytorch.org/docs/stable/data.html>
- PyTorch 多进程最佳实践（spawn / forkserver / CUDA）：<https://pytorch.org/docs/stable/notes/multiprocessing.html>
- OpenCV 安装包选择（每环境只装一个）：<https://pypi.org/project/opencv-python-headless/>
- 坐标取整差异：官方邮件列表「重叠子区域每通道最多差 1」<https://lists.andrew.cmu.edu/pipermail/openslide-users/2016-January/001175.html> · <https://github.com/openslide/openslide/issues/222> · <https://github.com/Bayer-Group/tiffslide/issues/51>

项目文档：[`OPENSLIDE_COMPAT.md`](https://github.com/khtao/fmg/blob/main/OPENSLIDE_COMPAT.md)（对齐表与核验口径）· [`OPTIMIZATION.md`](https://github.com/khtao/fmg/blob/main/OPTIMIZATION.md)（0.3.0 → 0.4.0 根因）· [`RELEASE.md`](https://github.com/khtao/fmg/blob/main/RELEASE.md)（发布手册）· `compat_report.md`（实测报告）

原项目：<https://github.com/khtao/fmg> · 作者 khtao · MIT
