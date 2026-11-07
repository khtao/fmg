# -*- coding: utf-8 -*-
"""Regression tests for fmg 0.4.

Covers:
  1. non-square images (width/height order bug in the original writer)
  2. region alignment, level>0 OpenSlide semantics, out-of-bounds padding
  3. pickling (spawn / DataLoader compatibility)
  4. forked workers inheriting an *already opened* slide (the crash case)
  5. reading files written by the original 0.3.0 code (backward compat)
  6. several slide objects / clones over one path sharing a single LMDB env
  7. the 'spawn' start method unpickling a slide in a fresh process

Run:  python selftest.py
"""
import importlib.util
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import time
import warnings

import cv2
import lmdb
import numpy as np

import fmg
import fmg.core

BASE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(BASE, '_selftest_data')
H, W = 768, 1024          # deliberately NON-square


def _install_eio_retry():
    """The sandbox's virtiofs mount sporadically returns EIO on mmap setup.

    This is an environment quirk, not a library issue: retry a handful of
    times and remove the half-created data.mdb before retrying a create.
    """
    real_open = lmdb.open

    def retrying_open(path, **kw):
        # py-lmdb defaults to create=True, so treat every non-readonly open as
        # a creating one and clear half-created files before retrying.
        creating = not bool(kw.get('readonly', False))
        for _ in range(200):
            try:
                return real_open(path, **kw)
            except lmdb.Error as exc:
                if 'Input/output' not in str(exc):
                    raise
                if creating:
                    for name in ('data.mdb', 'lock.mdb'):
                        try:
                            os.remove(os.path.join(path, name))
                        except OSError:
                            pass
                time.sleep(0.02)
        raise RuntimeError('LMDB open kept failing with EIO on %s' % path)

    fmg.core.lmdb.open = retrying_open


def make_image(h=H, w=W):
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[..., 0] = ((xx // 8) * 11 + (yy // 16) * 7) % 256
    img[..., 1] = ((yy // 8) * 9 + (xx // 32) * 5) % 256
    img[..., 2] = (((xx // 16) + (yy // 16)) * 17) % 256
    return img


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print('  ok  %s' % msg)


def test_single_process(img, path):
    print('[1] single-process correctness')
    slide = fmg.open_slide(path, patch_cache=64)
    check(tuple(slide.dimensions) == (W, H), 'dimensions == (width, height)')
    check(slide.level_count >= 2, 'pyramid has more than one level')

    out = np.asarray(slide.read_region((100, 50), 0, (512, 512)))
    ref = img[50:562, 100:612]
    check(out.shape == (512, 512, 3), 'read_region shape')
    check(np.array_equal(out, ref), 'unaligned region matches source exactly (lossless PNG)')

    out = np.asarray(slide.read_region((900, 700), 0, (256, 256)))
    check(out.shape == (256, 256, 3), 'out-of-bounds region shape')
    check(np.all(out[68:, 124:] == 0) and np.all(out[:68, 124:] == 0)
          and np.all(out[68:, :124] == 0), 'out-of-bounds area padded with black')

    lw, lh = slide.level_dimensions[1]
    ref_low = cv2.resize(img, (lw, lh), interpolation=cv2.INTER_AREA)
    low = np.asarray(slide.read_region((0, 0), 1, (256, 256)))
    check(np.array_equal(low, ref_low[:256, :256]),
          'level=1 uses level-0 location / level coordinates (OpenSlide semantics)')
    low2 = np.asarray(slide.read_region((200, 100), 1, (64, 64)))
    check(np.array_equal(low2, ref_low[50:114, 100:164]), 'level=1 with offset')

    clone = pickle.loads(pickle.dumps(slide))
    check(np.array_equal(np.asarray(clone.read_region((100, 50), 0, (64, 64))),
                         img[50:114, 100:164]), 'pickle/unpickle (spawn-safe)')

    with slide:
        pass
    check(slide._envs is None, 'context manager closes own handles')
    return slide


def test_shared_env(img, path):
    print('[1b] multiple slides / pickled clones over one path share one env')
    a = fmg.open_slide(path, patch_cache=0)
    b = fmg.open_slide(path, patch_cache=0)
    clone = pickle.loads(pickle.dumps(a))
    ref = img[50:114, 100:164]
    for obj, name in ((a, 'original'), (b, 'second open_slide'),
                      (clone, 'pickled clone')):
        got = np.asarray(obj.read_region((100, 50), 0, (64, 64)))
        check(np.array_equal(got, ref),
              '%s reads correct pixels while the others are open' % name)
    check(a._envs[0] is b._envs[0] is clone._envs[0],
          'all three share one process-local Environment')
    a.close()
    check(np.array_equal(np.asarray(b.read_region((100, 50), 0, (64, 64))), ref),
          'closing one slide does not invalidate the others')
    b.close()
    check(np.array_equal(np.asarray(clone.read_region((100, 50), 0, (64, 64))), ref),
          'last remaining slide still reads after the others close')
    clone.close()


def test_openslide_surface(path):
    print('[1c] OpenSlide API surface')
    slide = fmg.open_slide(path)
    check(isinstance(slide.level_dimensions, tuple)
          and all(isinstance(d, tuple) for d in slide.level_dimensions),
          'level_dimensions is a tuple of tuples')
    check(isinstance(slide.level_downsamples, tuple),
          'level_downsamples is a tuple')
    check(fmg.detect_format(path) == 'fmg', 'detect_format(.fmg) == "fmg"')
    check(fmg.detect_format(os.path.join(BASE, 'selftest.py')) is None,
          'detect_format(non-fmg) is None')
    check(slide.get_best_level_for_downsample(1.5) == 0
          and slide.get_best_level_for_downsample(2.0) == 1,
          'get_best_level_for_downsample follows OpenSlide C semantics')
    check('openslide.vendor' in slide.properties
          and 'openslide.level[0].width' in slide.properties,
          'properties exposes openslide.* keys')
    check(len(slide.associated_images) == 0,
          'associated_images is an empty mapping for numpy-written slides')
    check(slide.color_profile is None, 'color_profile is None')
    slide.set_cache(object())
    check(True, 'set_cache is a harmless no-op')
    thumb = slide.get_thumbnail((64, 64))
    check(thumb.mode == 'RGB' and max(thumb.size) <= 64, 'get_thumbnail works')
    slide.close()

    rgba = fmg.open_slide(path, rgba=True)
    img = rgba.read_region((10, 10), 0, (32, 32))
    check(img.mode == 'RGBA', 'rgba=True returns RGBA')
    arr = np.asarray(img)
    check(np.all(arr[..., 3] == 255), 'in-bounds alpha is 255')
    edge = np.asarray(rgba.read_region((W - 8, H - 8), 0, (32, 32)))
    check(np.all(edge[8:, 8:, 3] == 0), 'out-of-bounds alpha is 0')
    rgba.close()

    try:
        fmg.open_slide(path).read_region((0, 0), 999, (8, 8))
        raise AssertionError('expected OpenSlideError')
    except fmg.OpenSlideError:
        check(True, 'invalid level raises OpenSlideError')
    try:
        import openslide
        check(issubclass(fmg.FmgError, openslide.OpenSlideError),
              'FmgError is catchable as openslide.OpenSlideError')
    except ImportError:
        print('  skip openslide exception hierarchy (openslide not installed)')


def _worker(slide, queue, n, seed, img):
    try:
        rng = np.random.default_rng(seed)
        for _ in range(n):
            x = int(rng.integers(0, W - 64))
            y = int(rng.integers(0, H - 64))
            got = np.asarray(slide.read_region((x, y), 0, (64, 64)))
            if not np.array_equal(got, img[y:y + 64, x:x + 64]):
                raise AssertionError('data mismatch at (%d,%d)' % (x, y))
        queue.put(('ok', os.getpid(), slide._pid))
    except BaseException as exc:      # noqa: BLE001 - report worker crash to parent
        queue.put(('fail', os.getpid(), repr(exc)))


def test_fork(img, path):
    print('[2] forked workers inherit an ALREADY OPEN slide (crash reproducer)')
    parent = fmg.open_slide(path, patch_cache=64)
    _ = parent.read_region((0, 0), 0, (64, 64))   # env is open in the parent now
    check(parent._envs is not None and parent._pid == os.getpid(),
          'parent holds an open LMDB environment before fork')

    ctx = mp.get_context('fork')
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(parent, queue, 250, 1000 + i, img))
             for i in range(4)]
    for p in procs:
        p.start()

    # The parent keeps reading while the workers run: exercises concurrent
    # use of the same slide object from parent + 4 children.
    for i in range(200):
        _ = parent.read_region(((i * 7) % (W - 64), (i * 11) % (H - 64)), 0, (64, 64))

    results = [queue.get(timeout=180) for _ in procs]
    for p in procs:
        p.join(timeout=30)
    for p in procs:
        check(p.exitcode == 0, 'worker pid %s exited cleanly' % p.pid)
    for r in results:
        check(r[0] == 'ok', 'worker %s reported success' % r[1])
        check(r[1] == r[2], 'worker %s reopened the env under its own pid' % r[1])
    parent.close()


def _spawn_worker(slide, queue, img):
    try:
        got = np.asarray(slide.read_region((100, 50), 0, (64, 64)))
        queue.put(('ok', os.getpid(),
                   bool(np.array_equal(got, img[50:114, 100:164]))))
    except BaseException as exc:      # noqa: BLE001 - report worker crash to parent
        queue.put(('fail', os.getpid(), repr(exc)))


def test_spawn(img, path):
    print('[2b] spawn start method unpickles the slide in a fresh process')
    slide = fmg.open_slide(path, patch_cache=0)
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    proc = ctx.Process(target=_spawn_worker, args=(slide, queue, img))
    proc.start()
    result = queue.get(timeout=180)
    proc.join(timeout=30)
    check(proc.exitcode == 0, 'spawn worker exited cleanly')
    check(result[0] == 'ok' and result[2],
          'spawn worker unpickled the slide and read the correct pixels: %r' % (result,))
    slide.close()


def test_jpeg(path_jpg, img):
    print('[3] default JPEG writer smoke test')
    writer = fmg.create_slide(path_jpg)
    meta = writer.save_from_numpy(img[:256, :256], pyramid=False, mpp=0.25)
    check(meta['format_version'] == fmg.FORMAT_VERSION, 'format_version written')
    slide = fmg.open_slide(path_jpg)
    out = np.asarray(slide.read_region((0, 0), 0, (256, 256)))
    mae = np.abs(out.astype(np.int16) - img[:256, :256].astype(np.int16)).mean()
    check(mae < 12, 'JPEG round-trip MAE = %.2f (<12)' % mae)
    slide.close()


def test_legacy(path_legacy):
    print('[4] backward compatibility with files written by fmg 0.3.0')
    legacy_src = '/data/inputs/core.py'
    if not os.path.exists(legacy_src):
        print('  skip (original core.py not available)')
        return
    spec = importlib.util.spec_from_file_location('fmg_legacy_core', legacy_src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    square = np.ascontiguousarray(make_image(512, 512))
    mod.FastMemoryGraphics(path_legacy, 'w').save_from_numpy(
        square, ext='.png', params=[], mpp=0.25, pyramid=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        slide = fmg.open_slide(path_legacy, patch_cache=0)
    check(tuple(slide.dimensions) == (512, 512), 'legacy dimensions normalised to (w, h)')
    out = np.asarray(slide.read_region((0, 0), 0, (512, 512)))
    check(np.array_equal(out, square), 'legacy tiles decode correctly')
    slide.close()


def main():
    _install_eio_retry()
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    img = make_image()
    path = os.path.join(TMP, 'slide.fmg')

    t0 = time.time()
    meta = fmg.create_slide(path).save_from_numpy(
        img, ext='.png', pyramid=True, mpp=0.25, patch_size=256)
    print('write: %.2fs, levels=%s' % (time.time() - t0, meta['level_dimensions']))

    test_single_process(img, path)
    test_openslide_surface(path)
    test_shared_env(img, path)
    test_fork(img, path)
    test_spawn(img, path)
    test_jpeg(os.path.join(TMP, 'slide_jpg.fmg'), img)
    test_legacy(os.path.join(TMP, 'legacy.fmg'))

    print('\nALL TESTS PASSED')
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
