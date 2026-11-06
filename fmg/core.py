# -*- coding: utf-8 -*-
"""Fast Memory Graphics: a tiled, LMDB-backed whole-slide image.

Storage layout (unchanged, so existing .fmg files stay readable)
---------------------------------------------------------------
    <basename>.fmg                 JSON metadata
    <basename>/level0/            LMDB env, keys = str((x, y, x_max, y_max))
    <basename>/level1/
    ...

A tile is encoded with ``cv2.imencode`` and decoded with ``cv2.imdecode``.
The two are inverse operations, so the array that went in comes back out
with the same channel order (the library treats it as RGB at the API edge).

Concurrency contract
--------------------
* An ``Environment`` and its ``Transaction`` objects are never created in
  ``__init__``; they are opened lazily, once per process.
* If the object is inherited by a forked child (e.g. a PyTorch DataLoader
  worker), the inherited handles are dropped without being closed and a
  fresh environment is opened in the child.  Reusing a forked LMDB env is
  undefined behaviour and is the usual cause of sporadic SIGBUS/SIGSEGV.
* Every ``read_region`` uses its own short-lived read transaction, so the
  same slide object is safe to use from multiple threads of one process.
* Within one process, all slides pointing at the same directory share one
  refcounted read-only Environment: py-lmdb refuses to open the same path
  twice, and unpickling a slide in its parent process would otherwise fail.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import threading
import warnings
from collections import OrderedDict

import cv2
import lmdb
import numpy as np
from PIL import Image

__all__ = [
    'FmgError', 'CorruptTileError', 'AbstractSlide',
    'FastMemoryGraphics', 'configure', 'FORMAT_VERSION',
]

FORMAT_VERSION = 2
DEFAULT_PATCH_SIZE = 256
# Readers pass map_size=0, so LMDB reads the real value from the file's meta
# page (works for files written by 0.3.0 with a 1 TiB map too).  Writers get
# a size derived from the actual image instead of blindly mapping 1 TiB per
# level per process.
MIN_WRITE_MAP_SIZE = 1 << 28       # 256 MiB
MAX_WRITE_MAP_SIZE = 1 << 40       # 1 TiB, same as the original writer
DEFAULT_COMMIT_EVERY = 2048
SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}

_cv2_pid = None
_cv2_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Process-local registry of read-only LMDB environments.
#
# py-lmdb refuses to open the same path twice in one process ("The environment
# ... is already open in this process").  That happens as soon as two slide
# objects point at the same .fmg: a Dataset plus a pickled/unpickled clone,
# two Dataset instances over one slide, or a validation loader alongside the
# training loader.  Sharing ONE read-only Environment (with a fresh short
# transaction per read) is safe and avoids mapping the same file N times.
# ---------------------------------------------------------------------------
_env_registry = {}                 # (realpath, map_size) -> {'env': ..., 'refs': n}
_env_registry_pid = os.getpid()
_env_registry_lock = threading.RLock()


def _after_fork_in_child():
    """Drop inherited handles in the child; never close them here.

    ``mdb_env_close()`` in a forked child may clear the parent's reader slot
    / shared lock file, so the inherited Environment objects are simply
    forgotten.  The child opens its own on first use.  Rebuilding the locks
    also avoids a deadlock when another thread held one at fork time.
    """
    global _env_registry, _env_registry_pid, _env_registry_lock
    global _cv2_pid, _cv2_lock
    _env_registry = {}
    _env_registry_pid = os.getpid()
    _env_registry_lock = threading.RLock()
    _cv2_pid = None
    _cv2_lock = threading.Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _acquire_read_env(path, map_size):
    """Return a process-local read-only Environment shared by all slides."""
    global _env_registry_pid
    key = (os.path.realpath(path), int(map_size or 0))
    with _env_registry_lock:
        if _env_registry_pid != os.getpid():   # platforms without register_at_fork
            _env_registry.clear()
            _env_registry_pid = os.getpid()
        entry = _env_registry.get(key)
        if entry is not None:
            entry['refs'] += 1
            return entry['env']
        env = lmdb.open(
            path,
            subdir=True,
            readonly=True,
            create=False,
            lock=False,         # MDB_NOLOCK: no reader table, fork-safe
            readahead=False,    # WSI reads are random, prefetch hurts
            meminit=False,
            map_size=map_size or 0,   # 0 = use the size stored in meta
            max_spare_txns=8,        # makes begin() almost free
        )
        _env_registry[key] = {'env': env, 'refs': 1}
        return env


def _release_read_env(env):
    """Release one reference; close only when the last user goes away."""
    if env is None:
        return
    with _env_registry_lock:
        for key, entry in list(_env_registry.items()):
            if entry['env'] is env:
                entry['refs'] -= 1
                if entry['refs'] <= 0:
                    del _env_registry[key]
                    try:
                        env.close()
                    except Exception:
                        pass
                return
        # Not registered (e.g. inherited across fork): leaving the reference
        # alone is the safe choice, see _after_fork_in_child().


def _tune_cv2_for_worker():
    """Make OpenCV fork-safe, once per process.

    OpenCV's internal thread pool / OpenCL context do not survive ``fork()``.
    A child that inherits an initialised pool can deadlock or crash inside
    ``imdecode``.  ``setNumThreads(0)`` disables the pool; it costs nothing
    for single-image JPEG decoding.
    """
    global _cv2_pid
    pid = os.getpid()
    with _cv2_lock:
        if _cv2_pid == pid:
            return
        try:
            cv2.setNumThreads(0)
        except Exception:  # pragma: no cover - exotic OpenCV builds
            pass
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:  # pragma: no cover
            pass
        _cv2_pid = pid


def configure(num_threads=0, use_opencl=False):
    """Explicitly configure OpenCV for a DataLoader worker.

    Normally the library calls this automatically when it detects that it is
    running inside a forked child.  Call it yourself in ``worker_init_fn``
    when using the ``spawn`` start method, or when you want to control the
    setting from outside.
    """
    global _cv2_pid
    with _cv2_lock:
        try:
            cv2.setNumThreads(int(num_threads))
        except Exception:
            pass
        try:
            cv2.ocl.setUseOpenCL(bool(use_opencl))
        except Exception:
            pass
        _cv2_pid = os.getpid()
    return os.getpid()


class FmgError(Exception):
    """Base exception for this package."""


class CorruptTileError(FmgError):
    """A tile is missing, empty or cannot be decoded."""


def _tile_key(box):
    """Key format is part of the on-disk format; keep tuple repr."""
    return str(tuple(int(v) for v in box)).encode('ascii')


def _default_params(ext):
    if ext in ('.png',):
        return [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    return [int(cv2.IMWRITE_JPEG_QUALITY), 90]


class AbstractSlide:
    """Minimal OpenSlide-like interface."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        raise NotImplementedError

    @property
    def level_count(self):
        raise NotImplementedError

    @property
    def level_dimensions(self):
        """[(width, height), ...] in level coordinates."""
        raise NotImplementedError

    @property
    def dimensions(self):
        """(width, height) of level 0."""
        return self.level_dimensions[0]

    @property
    def level_downsamples(self):
        """[float, ...], one downsample factor per level."""
        raise NotImplementedError

    @property
    def properties(self):
        raise NotImplementedError

    def read_region(self, location, level, size):
        """Return a PIL.Image.

        location: (x, y) in the **level 0** reference frame (OpenSlide
                  semantics).
        level:    pyramid level.
        size:     (width, height) in **level** coordinates.
        """
        raise NotImplementedError

    def save_from_numpy(self, data, ext='.jpg', params=None, pyramid=False,
                        mpp=None, patch_size=DEFAULT_PATCH_SIZE,
                        commit_every=DEFAULT_COMMIT_EVERY, map_size=None):
        raise NotImplementedError

    def save_from_tiles(self, patches, locations, level, ext, params, txn=None):
        raise NotImplementedError


class FastMemoryGraphics(AbstractSlide):
    """A tiled whole-slide image stored in LMDB.

    Parameters
    ----------
    filename : str
        Path to the ``.fmg`` JSON file.
    mode : {'r', 'w'}
    patch_cache : int
        Per-process LRU cache size for decoded tiles (0 disables it).  Very
        effective for multi-epoch training, where every worker reads the
        same tiles again and again.
    map_size : int or None
        Explicit LMDB map size.  ``None`` (default) means: use the size
        stored in the file when reading, and derive a size from the image
        when writing.  No more blind 1 TiB mapping per level per process.
    strict : bool
        Raise ``CorruptTileError`` on a missing/broken tile instead of
        painting that tile black and warning once.
    """

    def __init__(self, filename, mode='r', *, patch_cache=128, map_size=None,
                 strict=False, auto_worker_tuning=True):
        if mode not in ('r', 'w'):
            raise ValueError("mode must be 'r' or 'w'")
        self._filename = os.fspath(filename)
        self._mode = mode
        self._data_dir = os.path.splitext(self._filename)[0]  # same as old [:-4] for '.fmg'
        self._patch_cache_size = max(0, int(patch_cache))
        self._map_size = int(map_size) if map_size else None
        self._strict = bool(strict)
        self._auto_worker_tuning = bool(auto_worker_tuning)

        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self._envs = None            # {level: lmdb.Environment} owned by THIS process
        self._pid = None             # pid that owns self._envs
        self._creator_pid = os.getpid()   # survives fork(), not pickling
        self._closed = False
        self._missing_warned = False

        self._osr = {}
        self._stored_dims = []       # (grid_x, grid_y) as written on disk
        self._public_dims = []       # (width, height) exposed to callers
        self._downsamples = []
        self._legacy = False

        if mode == 'r':
            self._load_metadata()

    # ------------------------------------------------------------------ #
    # metadata
    # ------------------------------------------------------------------ #
    def _load_metadata(self):
        # 'with' fixes the leaked file descriptor of the original code.
        with open(self._filename, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)
        if not isinstance(meta, dict) or 'level_dimensions' not in meta:
            raise FmgError('not a valid .fmg metadata file: %s' % self._filename)
        self._osr = meta
        self._legacy = int(meta.get('format_version', 1)) < FORMAT_VERSION
        self._stored_dims = [tuple(int(v) for v in d)
                             for d in meta['level_dimensions']]
        if any(len(d) != 2 or d[0] <= 0 or d[1] <= 0 for d in self._stored_dims):
            raise FmgError('invalid level_dimensions in %s' % self._filename)
        # Legacy files stored [height, width] AND used that order as the tile
        # grid, so the grid itself is fine; only the public (w, h) view is
        # reversed.  New files store (width, height) == grid (x, y).
        if self._legacy:
            self._public_dims = [(d[1], d[0]) for d in self._stored_dims]
            warnings.warn(
                '%s uses the legacy format (level_dimensions stored as '
                '[height, width] and non-square images may miss tiles). '
                'Re-write the slide with save_from_numpy() to upgrade.'
                % self._filename, RuntimeWarning, stacklevel=2)
        else:
            self._public_dims = list(self._stored_dims)

        raw_ds = meta.get('level_downsamples') or [1.0] * len(self._stored_dims)
        self._downsamples = []
        for v in raw_ds:
            # legacy: [[dx, dy], ...]; new: [ds, ...]
            self._downsamples.append(float(v[0] if isinstance(v, (list, tuple)) else v))
        self._downsamples = (self._downsamples + [1.0] * len(self._stored_dims))[:len(self._stored_dims)]
        for i, d in enumerate(self._downsamples):
            if d <= 0 or not math.isfinite(d):
                self._downsamples[i] = 1.0

        self._osr.setdefault('patch_size', DEFAULT_PATCH_SIZE)
        self._osr.setdefault('ext', '.jpg')
        self._osr.setdefault('params', _default_params(self._osr['ext']))
        self._osr.setdefault('mpp', None)
        self._osr.setdefault('level_count', len(self._stored_dims))

    def __repr__(self):
        return f'{self.__class__.__name__}({self._filename!r}, mode={self._mode!r})'

    @property
    def level_count(self):
        return len(self._stored_dims)

    @property
    def level_dimensions(self):
        return [list(d) for d in self._public_dims]

    @property
    def dimensions(self):
        return self._public_dims[0]

    @property
    def level_downsamples(self):
        return list(self._downsamples)

    @property
    def properties(self):
        out = dict(self._osr)
        out['level_dimensions'] = self.level_dimensions
        out['dimension'] = list(self.dimensions)
        out['level_downsamples'] = self.level_downsamples
        return out

    @property
    def patch_size(self):
        return int(self._osr.get('patch_size', DEFAULT_PATCH_SIZE))

    # ------------------------------------------------------------------ #
    # process / thread safe environment management
    # ------------------------------------------------------------------ #
    def _ensure_open(self, level):
        """Return an LMDB Environment owned by the *current* process.

        Only the requested level is opened, on first use.  A forked child
        discards inherited handles and opens its own; this is the single most
        important fix for DataLoader crashes.
        """
        if self._mode != 'r':
            raise FmgError('slide is open for writing, not reading')
        if self._closed:
            raise FmgError('slide is closed')
        pid = os.getpid()
        is_forked_child = (self._creator_pid is not None
                           and self._creator_pid != pid)
        with self._lock:
            if is_forked_child and self._auto_worker_tuning:
                # Detected even when the parent never opened an env (lazy
                # case): OpenCV's inherited thread pool is still unsafe.
                _tune_cv2_for_worker()
            if self._envs is None or self._pid != pid:
                if self._envs is not None:
                    # NEVER close an inherited env: mdb_env_close() in the
                    # child may clear the parent's reader slot / shared lock
                    # file.  Dropping the reference alone is safe because we
                    # open with lock=False (no reader table is used at all).
                    self._cache.clear()
                self._envs = {}
                self._pid = pid
            env = self._envs.get(level)
            if env is not None:
                return env
            path = os.path.join(self._data_dir, 'level' + str(level))
            env = _acquire_read_env(path, self._map_size)
            self._envs[level] = env
            return env

    def close(self):
        """Idempotent.  Only closes handles owned by the current process."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            envs, owner = self._envs, self._pid
            self._envs = None
            self._cache.clear()
        if not envs or owner != os.getpid():
            return
        for env in envs.values():
            _release_read_env(env)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # pickling: a forked/pickled slide must not carry an LMDB Environment.
    # The clone reopens on first read; in the SAME process that read hits the
    # shared registry, in a new process (spawn) it opens its own env.
    # ------------------------------------------------------------------ #
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_envs'] = None
        state['_pid'] = None
        state['_creator_pid'] = None   # unpickled object is native to its process
        state['_lock'] = None
        state['_cache'] = OrderedDict()
        state['_closed'] = False
        state['_missing_warned'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self._envs = None
        self._pid = None
        self._creator_pid = os.getpid()
        self._closed = False
        self._missing_warned = False

    # ------------------------------------------------------------------ #
    # tile cache
    # ------------------------------------------------------------------ #
    def _cache_get(self, key):
        if not self._patch_cache_size:
            return None
        with self._lock:
            tile = self._cache.get(key)
            if tile is not None:
                self._cache.move_to_end(key)
            return tile

    def _cache_put(self, key, tile):
        if not self._patch_cache_size:
            return
        with self._lock:
            self._cache[key] = tile
            self._cache.move_to_end(key)
            while len(self._cache) > self._patch_cache_size:
                self._cache.popitem(last=False)

    def _broken(self, level, box, reason):
        msg = 'level %d tile %s: %s' % (level, tuple(box), reason)
        if self._strict:
            raise CorruptTileError(msg)
        if not self._missing_warned:
            warnings.warn(msg + ' (painted black)', RuntimeWarning, stacklevel=3)
            self._missing_warned = True

    # ------------------------------------------------------------------ #
    # reading
    # ------------------------------------------------------------------ #
    def _read_tile(self, level, box, txn):
        cache_key = b'%d/' % level + _tile_key(box)
        tile = self._cache_get(cache_key)
        if tile is not None:
            return tile

        raw = txn.get(key=_tile_key(box))
        if raw is None or len(raw) == 0:
            self._broken(level, box, 'missing or empty tile')
            return None
        try:
            arr = np.frombuffer(raw, dtype=np.uint8)
            tile = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:  # pragma: no cover - defensive
            self._broken(level, box, 'decode failed: %r' % (exc,))
            return None
        if tile is None or tile.ndim != 3 or tile.shape[2] != 3:
            self._broken(level, box, 'invalid decoded tile')
            return None
        self._cache_put(cache_key, tile)
        return tile

    def read_region(self, location, level, size):
        """Return a PIL.Image of ``size`` (width, height) in level coordinates.

        ``location`` is given in the level-0 frame, as in OpenSlide.  Regions
        outside the slide are padded with black instead of raising.
        """
        if self._mode != 'r':
            raise FmgError('slide is open for writing, not reading')
        if not 0 <= level < self.level_count:
            raise ValueError('level %d out of range [0, %d)' % (level, self.level_count))
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0:
            raise ValueError('size must be positive, got %r' % (size,))
        x0, y0 = int(location[0]), int(location[1])

        env = self._ensure_open(level)

        ds = self._downsamples[level]
        lx0 = int(round(x0 / ds))
        ly0 = int(round(y0 / ds))
        lx1, ly1 = lx0 + width, ly0 + height

        step = self.patch_size
        grid_x, grid_y = self._stored_dims[level]
        out = np.zeros((height, width, 3), dtype=np.uint8)

        x_first = math.floor(lx0 / step) * step
        y_first = math.floor(ly0 / step) * step
        x_last = min(grid_x, math.ceil(lx1 / step) * step)
        y_last = min(grid_y, math.ceil(ly1 / step) * step)

        with env.begin(buffers=True) as txn:
            for ty in range(y_first, y_last, step):
                if ty < 0:
                    continue
                ty_max = min(ty + step, grid_y)
                iy0, iy1 = max(ty, ly0), min(ty_max, ly1)
                if iy0 >= iy1:
                    continue
                for tx in range(x_first, x_last, step):
                    if tx < 0:
                        continue
                    tx_max = min(tx + step, grid_x)
                    ix0, ix1 = max(tx, lx0), min(tx_max, lx1)
                    if ix0 >= ix1:
                        continue
                    box = (tx, ty, tx_max, ty_max)
                    tile = self._read_tile(level, box, txn)
                    if tile is None:
                        continue
                    src = tile[iy0 - ty:iy1 - ty, ix0 - tx:ix1 - tx]
                    dst = out[iy0 - ly0:iy1 - ly0, ix0 - lx0:ix1 - lx0]
                    if src.shape != dst.shape:
                        # Damaged / mismatched legacy tile; do not crash a
                        # whole training job over one bad patch.
                        self._broken(level, box, 'tile shape %s != %s'
                                     % (src.shape, dst.shape))
                        continue
                    dst[:] = src
        return Image.fromarray(out)

    # ------------------------------------------------------------------ #
    # writing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_uint8_hwc(data):
        arr = np.asarray(data)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError('data must be HWC or HW, got shape %s' % (arr.shape,))
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] != 3:
            raise ValueError('data must have 1, 3 or 4 channels, got %d' % arr.shape[2])
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _put_tiles(txn, patches, locations, ext, params):
        for patch, box in zip(patches, locations):
            arr = np.ascontiguousarray(FastMemoryGraphics._as_uint8_hwc(patch))
            ok, buff = cv2.imencode(ext, arr, params)
            if not ok or buff is None or buff.size == 0:
                raise FmgError('cv2.imencode failed for tile %s' % (tuple(box),))
            # memoryview avoids the extra bytes() copy of the original code.
            txn.put(key=_tile_key(box), value=memoryview(buff))

    @staticmethod
    def _estimate_map_size(dims):
        raw = sum(w * h for w, h in dims) * 3       # 3 channels, uint8
        return max(MIN_WRITE_MAP_SIZE,
                   min(MAX_WRITE_MAP_SIZE, int(raw * 3 + (1 << 20) * len(dims))))

    def _write_env(self, level, dims, map_size):
        path = os.path.join(self._data_dir, 'level' + str(level))
        env = lmdb.open(path, subdir=True, create=True,
                        map_size=map_size, max_spare_txns=1)
        txn = env.begin(write=True)
        txn.put(b'patch_size', str(self._osr['patch_size']).encode())
        txn.put(b'mpp', str(self._osr['mpp']).encode())
        txn.put(b'dimension', str(list(dims)).encode())
        txn.commit()
        return env

    def save_from_numpy(self, data, ext='.jpg', params=None, pyramid=False,
                        mpp=None, patch_size=DEFAULT_PATCH_SIZE,
                        commit_every=DEFAULT_COMMIT_EVERY, map_size=None):
        """Write a whole slide.

        ``level_dimensions`` is now stored as (width, height) and the tile
        grid iterates x over width and y over height -- the original code
        mixed the two up, which silently corrupted every non-square image.
        """
        if self._mode != 'w':
            raise FmgError('slide is open for reading, not writing')
        ext = ext.lower()
        if ext not in SUPPORTED_EXT:
            raise ValueError('unsupported ext %r, choose from %s' % (ext, sorted(SUPPORTED_EXT)))
        params = [int(p) for p in params] if params is not None else _default_params(ext)
        patch_size = int(patch_size)
        if patch_size <= 0:
            raise ValueError('patch_size must be positive')

        level0 = self._as_uint8_hwc(data)
        h, w, _ = level0.shape

        dims = [(w, h)]
        downsamples = [1.0]
        nw, nh = w, h
        # Only the *sizes* are planned here; each level image is generated
        # once, directly from level 0, inside the write loop below.  The old
        # code kept a list of full copies of every level in RAM.
        while pyramid and max(nw, nh) > 512 and len(dims) < 12:
            nw, nh = max(1, nw // 2), max(1, nh // 2)
            dims.append((nw, nh))
            downsamples.append(downsamples[-1] * 2.0)

        map_size = int(map_size) if map_size else (
            self._map_size or self._estimate_map_size(dims))
        map_size = max(MIN_WRITE_MAP_SIZE, map_size)

        properties = {
            'format_version': FORMAT_VERSION,
            'map_size': map_size,
            'grid_axes': 'xy',           # level_dimensions == tile grid (x, y)
            'level_count': len(dims),
            'level_dimensions': [list(d) for d in dims],
            'dimension': list(dims[0]),
            'level_downsamples': downsamples,
            'mpp': float(mpp) if mpp is not None else None,
            'ext': ext,
            'params': params,
            'patch_size': patch_size,
            'channels': 3,
            'tile_codec': 'opencv',
        }
        self._osr = properties
        self._stored_dims = [tuple(d) for d in dims]
        self._public_dims = [tuple(d) for d in dims]
        self._downsamples = list(downsamples)
        self._legacy = False

        if os.path.exists(self._data_dir):
            shutil.rmtree(self._data_dir)
        os.makedirs(self._data_dir)

        for level, (gw, gh) in enumerate(dims):
            layer = (level0 if level == 0 else
                     cv2.resize(level0, (gw, gh), interpolation=cv2.INTER_AREA))
            env = self._write_env(level, (gw, gh), map_size)
            try:
                txn = env.begin(write=True)
                written = 0
                for x in range(0, gw, patch_size):
                    for y in range(0, gh, patch_size):
                        x_max, y_max = min(x + patch_size, gw), min(y + patch_size, gh)
                        arr = np.ascontiguousarray(layer[y:y_max, x:x_max, :])
                        ok, buff = cv2.imencode(ext, arr, params)
                        if not ok or buff is None or buff.size == 0:
                            raise FmgError('cv2.imencode failed for tile (%d,%d,%d,%d)'
                                           % (x, y, x_max, y_max))
                        txn.put(key=_tile_key((x, y, x_max, y_max)), value=memoryview(buff))
                        written += 1
                        if written % commit_every == 0:
                            txn.commit()
                            txn = env.begin(write=True)
                txn.commit()
                env.sync()
            finally:
                env.close()

        # Atomic metadata write: never leave a .fmg pointing at half-written data.
        tmp = self._filename + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(properties, fh, separators=(',', ':'))
        os.replace(tmp, self._filename)
        return properties

    def save_from_tiles(self, patches, locations, level, ext, params=None, txn=None):
        """Append tiles to ``level``.

        Existing callers work unchanged.  A write transaction may be supplied
        (``txn=``) so a caller can batch thousands of tiles into one commit.
        """
        if self._mode != 'w':
            raise FmgError('slide is open for reading, not writing')
        if not 0 <= level < self.level_count:
            raise ValueError('level %d out of range' % level)
        ext = ext.lower()
        if ext not in SUPPORTED_EXT:
            raise ValueError('unsupported ext %r' % ext)
        params = [int(p) for p in params] if params is not None else _default_params(ext)
        if txn is not None:
            self._put_tiles(txn, patches, locations, ext, params)
            return
        path = os.path.join(self._data_dir, 'level' + str(level))
        map_size = self._map_size or self._estimate_map_size(self._stored_dims)
        env = lmdb.open(path, subdir=True, create=True,
                        map_size=map_size, max_spare_txns=1)
        try:
            with env.begin(write=True) as t:
                self._put_tiles(t, patches, locations, ext, params)
            env.sync()
        finally:
            env.close()
