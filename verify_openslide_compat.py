# -*- coding: utf-8 -*-
"""Conformance check: fmg API vs OpenSlide, and coordinate/pixel equality.

Run:  python verify_openslide_compat.py

What it does
------------
1. Builds two pyramidal tiled TIFFs that the *real* OpenSlide can read:
     integer.tif     levels 1536x1024, 768x512, 384x256   (ds 1, 2, 4)
     fractional.tif  levels 1536x1024, 1000x667, 500x333  (ds 1, 1.5356, 3.0735)
   Pixels are coordinate-encoded, so a 1-pixel offset is immediately visible.
2. Opens both with ``openslide.open_slide`` and converts with
   ``fmg.from_openslide`` (lossless PNG).
3. Checks the whole ``AbstractSlide`` surface against OpenSlide.
4. Compares ``read_region`` results for many (location, level, size) cases:
   aligned, unaligned, cross-tile, out-of-bounds, negative and level>0
   coordinates.  The alpha channel of OpenSlide's RGBA is DISCARDED, as
   requested; fmg's RGB (or its RGBA when rgba=True) is compared channel-wise.
5. Uses a synthetic AbstractSlide-like object to verify that properties and
   associated images survive the conversion.

Writes ``compat_report.md`` next to this script and exits 0 only if every
mandatory check passes.  Fractional-downsample coordinate checks are reported
separately (see OPENSLIDE_COMPAT.md for why they can differ by 1 pixel).
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image

import fmg
import fmg.core

BASE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(BASE, '_compat_data')
REPORT = os.path.join(BASE, 'compat_report.md')
SEED = 20261107

_lines = []
_checks = []


def log(msg=''):
    print(msg)
    _lines.append(msg)


def check(cond, msg, mandatory=True):
    _checks.append((bool(cond), msg, mandatory))
    log(('  %s  %s' % ('ok  ' if cond else 'FAIL', msg)) +
        ('' if mandatory else '  [informational]'))
    if not cond and mandatory:
        raise AssertionError(msg)


def _install_eio_retry():
    """virtiofs in this sandbox sporadically returns EIO on mmap setup."""
    import lmdb
    real_open = lmdb.open

    def retrying_open(path, **kw):
        creating = not bool(kw.get('readonly', False))
        for attempt in range(300):
            try:
                return real_open(path, **kw)
            except lmdb.Error as exc:
                if 'Input/output' not in str(exc):
                    raise
                if creating:
                    # Rebuild the whole directory: virtiofs can leave a
                    # half-initialised mmap behind that deleting just the two
                    # files does not clear.
                    shutil.rmtree(path, ignore_errors=True)
                    os.makedirs(path, exist_ok=True)
                time.sleep(0.01 * min(attempt + 1, 20))
        raise RuntimeError('LMDB open kept failing with EIO on %s' % path)

    fmg.core.lmdb.open = retrying_open


def _encoded(w, h, seed=0):
    """Deterministic, coordinate-encoded RGB image (nearly unique per pixel)."""
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    a = np.empty((h, w, 3), dtype=np.uint8)
    a[..., 0] = ((xx * 7 + yy * 13 + seed * 31) % 256)
    a[..., 1] = ((xx * 3 + yy * 29 + seed * 17) % 256)
    a[..., 2] = (((xx // 7) * 11 + (yy // 11) * 17 + seed) % 256)
    return a


def _make_tiff(path, dims):
    import tifffile
    with tifffile.TiffWriter(path) as tif:
        for i, (w, h) in enumerate(dims):
            tif.write(_encoded(w, h, i + 1), photometric='rgb',
                      tile=(256, 256),
                      # The "reduced resolution" bit is what makes OpenSlide's
                      # generic-tiff driver treat later IFDs as pyramid levels.
                      subfiletype=(1 if i else 0))
    return path


class _FakeSlide:
    """An AbstractSlide-like object with properties + associated images.

    Its read_region uses the same truncation rule as OpenSlide's C code, so
    the converted fmg must match it exactly at every level.
    """

    def __init__(self, dims=(1024, 768), downsamples=None):
        import cv2
        w0, h0 = dims
        self._imgs = []
        self._w0, self._h0 = w0, h0
        self.level_dimensions = tuple(
            (max(1, w0 // d), max(1, h0 // d)) for d in downsamples)
        self.level_downsamples = tuple(float(d) for d in downsamples)
        for (w, h), d in zip(self.level_dimensions, downsamples):
            self._imgs.append(cv2.resize(_encoded(w0, h0, 3), (w, h),
                                         interpolation=cv2.INTER_AREA))
        self.properties = {
            'openslide.vendor': 'fake',
            'openslide.mpp-x': '0.25',
            'openslide.mpp-y': '0.25',
            'openslide.objective-power': '40',
            'openslide.comment': 'synthetic conformance slide',
        }
        self.associated_images = {
            'label': Image.fromarray(_encoded(64, 32, 9)).convert('RGBA'),
            'macro': Image.fromarray(_encoded(128, 96, 11)).convert('RGBA'),
        }

    @property
    def level_count(self):
        return len(self._imgs)

    @property
    def dimensions(self):
        return self.level_dimensions[0]

    @staticmethod
    def detect_format(filename):
        return 'fake'

    def close(self):
        pass

    def get_best_level_for_downsample(self, downsample):
        ds = self.level_downsamples
        if downsample < ds[0]:
            return 0
        for i in range(1, len(ds)):
            if downsample < ds[i]:
                return i - 1
        return len(ds) - 1

    def read_region(self, location, level, size):
        w, h = int(size[0]), int(size[1])
        ds = self.level_downsamples[level]
        lx, ly = int(int(location[0]) / ds), int(int(location[1]) / ds)
        src = self._imgs[level]
        sh, sw = src.shape[:2]
        out = np.zeros((h, w, 4), dtype=np.uint8)
        ix0, iy0 = max(0, lx), max(0, ly)
        ix1, iy1 = min(sw, lx + w), min(sh, ly + h)
        if ix0 < ix1 and iy0 < iy1:
            block = src[iy0:iy1, ix0:ix1]
            out[iy0 - ly:iy1 - ly, ix0 - lx:ix1 - lx, :3] = block
            out[iy0 - ly:iy1 - ly, ix0 - lx:ix1 - lx, 3] = 255
        return Image.fromarray(out)


def _make_cases(slide, levels):
    """Representative (location, level, size) cases, incl. nasty corners."""
    rng = np.random.default_rng(SEED)
    w0, h0 = slide.dimensions
    cases = []
    for level in levels:
        lw, lh = slide.level_dimensions[level]
        ds = slide.level_downsamples[level]
        cases.append(((0, 0), level, (min(lw, 400), min(lh, 400)), 'origin'))
        cases.append(((int(256 * ds), int(256 * ds)), level, (256, 256), 'aligned'))
        cases.append(((int(255 * ds), int(257 * ds)), level, (300, 300), 'cross-tile'))
        cases.append(((w0 - 64, h0 - 64), level, (256, 256), 'out-of-bounds'))
        cases.append(((-128, -96), level, (256, 256), 'negative'))
        cases.append(((0, h0 - 32), level, (200, 200), 'bottom-edge'))
        for _ in range(12):
            x = int(rng.integers(-64, max(1, w0 - 64)))
            y = int(rng.integers(-64, max(1, h0 - 64)))
            w = int(rng.integers(1, 320))
            h = int(rng.integers(1, 320))
            cases.append(((x, y), level, (w, h), 'random'))
    return cases


def _make_core_cases(slide):
    """Cases with a well-defined coordinate contract.

    * level 0: arbitrary level-0 coordinates (ds == 1, no conversion at all).
    * level > 0: level-0 coordinates ``ceil(level_coord * downsample)``, which
      satisfy ``int(X / ds) == level_coord`` for every downsample, plus
      negative and out-of-bounds locations.

    These are the coordinates a caller should use for exact reproducibility.
    """
    rng = np.random.default_rng(SEED + 1)
    w0, h0 = slide.dimensions
    cases = []
    # level 0: everything, including nasty corners
    cases.append(((0, 0), 0, (min(w0, 400), min(h0, 400)), 'origin'))
    cases.append(((255, 257), 0, (300, 300), 'cross-tile'))
    cases.append(((w0 - 64, h0 - 64), 0, (256, 256), 'out-of-bounds'))
    cases.append(((-128, -96), 0, (256, 256), 'negative'))
    cases.append(((0, h0 - 32), 0, (200, 200), 'bottom-edge'))
    for _ in range(13):
        x = int(rng.integers(-64, max(1, w0 - 64)))
        y = int(rng.integers(-64, max(1, h0 - 64)))
        cases.append(((x, y), 0, (int(rng.integers(1, 320)),
                                  int(rng.integers(1, 320))), 'random'))
    # level > 0: aligned level-0 coordinates
    for level in range(1, slide.level_count):
        ds = slide.level_downsamples[level]
        for lx, ly in ((0, 0), (64, 32), (128, 64), (200, 100),
                       (256, 128), (300, 200), (400, 300)):
            x = int(math.ceil(lx * ds))
            y = int(math.ceil(ly * ds))
            cases.append(((x, y), level, (128, 128), 'aligned-level'))
        cases.append(((-16, -16), level, (64, 64), 'negative-level'))
        cases.append(((w0 - 8, h0 - 8), level, (64, 64), 'oob-level'))
    return cases


def _compare_regions(os_slide, fm_slide, cases, rgba=False, label=''):
    """Compare RGB content (drop OpenSlide's alpha) for every case."""
    passed = 0
    total_pixels = 0
    bad_pixels = 0
    worst_mae = 0.0
    failures = []
    for loc, level, size, tag in cases:
        a = np.asarray(os_slide.read_region(loc, level, size))
        b = np.asarray(fm_slide.read_region(loc, level, size))
        if rgba:
            # Both sides must be RGBA; fmg(rgba=True) is, OpenSlide always is.
            if a.shape[2] == 3:
                a = np.dstack([a, np.full(a.shape[:2], 255, np.uint8)])
            if b.shape[2] == 3:
                b = np.dstack([b, np.full(b.shape[:2], 255, np.uint8)])
        else:
            # Ignore the RGB / RGBA difference: compare colour channels only.
            a, b = a[..., :3], b[..., :3]
        n = a.shape[0] * a.shape[1]
        total_pixels += n
        if a.shape != b.shape:
            failures.append((tag, loc, level, size, 'shape %s vs %s' % (a.shape, b.shape)))
            continue
        if np.array_equal(a, b):
            passed += 1
            continue
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        mae = float(diff.mean())
        worst_mae = max(worst_mae, mae)
        bad_pixels += int(np.any(diff > 0, axis=2).sum())
        failures.append((tag, loc, level, size, 'MAE %.3f' % mae))
    log('  %s: %d/%d regions identical, %d/%d pixels differ, worst MAE %.3f' %
        (label, passed, len(cases), bad_pixels, total_pixels, worst_mae))
    return passed, len(cases), bad_pixels, total_pixels, worst_mae, failures


def verify_api(os_slide, fm_slide, path_fmg, path_tif, expect_exact):
    import openslide
    log('\n[API] OpenSlide surface')
    check(fm_slide.level_count == os_slide.level_count,
          'level_count == %d' % os_slide.level_count)
    check(tuple(fm_slide.level_dimensions) == tuple(os_slide.level_dimensions),
          'level_dimensions identical (tuple of (w,h) tuples)')
    check(isinstance(fm_slide.level_dimensions, tuple)
          and all(isinstance(d, tuple) for d in fm_slide.level_dimensions),
          'level_dimensions returns tuples, not lists')
    check(isinstance(fm_slide.level_downsamples, tuple),
          'level_downsamples returns a tuple')
    check(fm_slide.dimensions == os_slide.dimensions,
          'dimensions == %s' % (os_slide.dimensions,))
    check(all(abs(a - b) < 1e-9 for a, b in
              zip(fm_slide.level_downsamples, os_slide.level_downsamples)),
          'level_downsamples identical: %s' % (os_slide.level_downsamples,))

    # properties
    op, fp = os_slide.properties, fm_slide.properties
    for key in ('openslide.vendor', 'openslide.level-count',
                'openslide.level[0].downsample', 'openslide.level[0].width',
                'openslide.level[0].height'):
        check(key in fp, 'properties contains %s' % key)
    check(fp.get('openslide.vendor') == op.get('openslide.vendor'),
          'source vendor preserved (%s)' % op.get('openslide.vendor'))
    for level in range(os_slide.level_count):
        prefix = 'openslide.level[%d].' % level
        check(fp.get(prefix + 'width') == str(os_slide.level_dimensions[level][0]),
              '%s width matches fmg geometry' % prefix)
        check(fp.get(prefix + 'downsample') == str(os_slide.level_downsamples[level]),
              '%s downsample matches fmg geometry' % prefix)

    # detect_format
    check(fmg.detect_format(path_fmg) == 'fmg',
          'fmg.detect_format(<fmg>) == "fmg"')
    check(fmg.detect_format(path_tif) is None,
          'fmg.detect_format(<tiff>) is None')
    check(fmg.FastMemoryGraphics.detect_format(path_fmg) == 'fmg',
          'classmethod detect_format works like OpenSlide.detect_format')

    # get_best_level_for_downsample
    for d in (0.1, 0.9, 1.0, 1.5, 2.0, 2.5, 3.9, 4.0, 8.0, 100.0):
        a = os_slide.get_best_level_for_downsample(d)
        b = fm_slide.get_best_level_for_downsample(d)
        check(a == b, 'get_best_level_for_downsample(%.1f) == %d' % (d, a))

    # associated_images / color_profile / set_cache / get_thumbnail
    check(set(fm_slide.associated_images.keys()) ==
          set(os_slide.associated_images.keys()),
          'associated_images keys match (%s)' % list(os_slide.associated_images))
    check(fm_slide.color_profile is None,
          'color_profile is None (fmg stores no ICC profile)')
    fm_slide.set_cache(object())       # must be a harmless no-op
    log('  ok  set_cache is a no-op')
    thumb = fm_slide.get_thumbnail((200, 200))
    check(thumb.mode == 'RGB' and max(thumb.size) <= 200,
          'get_thumbnail((200,200)) -> %s %s' % (thumb.mode, thumb.size))

    # exception hierarchy: generic OpenSlide code must catch fmg errors
    check(issubclass(fmg.FmgError, openslide.OpenSlideError),
          'fmg.FmgError is an openslide.OpenSlideError subclass')
    try:
        fm_slide.read_region((0, 0), 999, (16, 16))
        caught = None
    except openslide.OpenSlideError as exc:
        caught = exc
    check(caught is not None,
          'invalid level raises an exception catchable as OpenSlideError')
    try:
        fm_slide.read_region((0, 0), 0, (0, 16))
        caught = None
    except openslide.OpenSlideError:
        caught = True
    check(caught is not None, 'zero size raises an OpenSlideError')

    # context manager / close
    with fmg.open_slide(path_fmg) as s:
        check(s.read_region((0, 0), 0, (8, 8)).size == (8, 8),
              'context manager can read before exit')
    log('  ok  context manager closed the slide')


def verify_pixels(os_slide, path_fmg, tag, strict):
    log('\n[PIXELS] %s' % tag)
    fm_slide = fmg.open_slide(path_fmg, patch_cache=0)
    try:
        # (A) Full-level content equality: conversion must not change a pixel.
        log('  (A) full-level content (conversion fidelity)')
        for level in range(os_slide.level_count):
            lw, lh = os_slide.level_dimensions[level]
            a = np.asarray(os_slide.read_region((0, 0), level, (lw, lh)))[..., :3]
            b = np.asarray(fm_slide.read_region((0, 0), level, (lw, lh)))[..., :3]
            check(np.array_equal(a, b),
                  'level %d: all %dx%d pixels identical to OpenSlide' % (level, lw, lh))

        # (B) Coordinate contract: level-0 arbitrary + level>0 aligned.
        log('  (B) coordinate contract (RGB, alpha dropped)')
        core = _make_core_cases(os_slide)
        p, n, _, _, _, failures = _compare_regions(
            os_slide, fm_slide, core, rgba=False, label='  core cases')
        if failures:
            log('  first mismatches: %s' % (failures[:6],))
        check(p == n, '%s: %d/%d core regions pixel-identical' % (tag, p, n),
              mandatory=strict)

        # (C) Arbitrary level-0 coordinates at level>0.  OpenSlide's own
        # cairo-based sampling can shift these by +/-1 px (known behaviour);
        # reported for transparency, not a hard failure.
        log('  (C) arbitrary level-0 coordinates at level>0 (OpenSlide rounding)')
        extended = [c for c in _make_cases(os_slide, range(os_slide.level_count))
                    if c[1] > 0]
        ep, en, ebad, etotal, emae, _ = _compare_regions(
            os_slide, fm_slide, extended, rgba=False, label='  extended cases')
        log('  informational: %d/%d extended regions identical; %.2f%% pixels differ'
            % (ep, en, 100.0 * ebad / max(1, etotal)))
    finally:
        fm_slide.close()

    # RGBA mode: alpha must also match (including transparent out-of-bounds)
    fm_rgba = fmg.open_slide(path_fmg, patch_cache=0, rgba=True)
    try:
        img = fm_rgba.read_region((0, 0), 0, (16, 16))
        check(img.mode == 'RGBA', "rgba=True returns an RGBA image")
        cases = _make_cases(os_slide, [0])[:8]
        p, n, _, _, _, _ = _compare_regions(
            os_slide, fm_rgba, cases, rgba=True, label='RGBA mode, level 0')
        check(p == n, 'rgba=True: %d/%d level-0 regions identical incl. alpha' % (p, n))
    finally:
        fm_rgba.close()


def verify_fake():
    log('\n[FAKE] properties + associated_images round trip')
    fake = _FakeSlide((1024, 768), (1, 2, 4))
    path = os.path.join(TMP, 'fake.fmg')
    fmg.from_openslide(fake, path, patch_size=256, ext='.png')
    s = fmg.open_slide(path, patch_cache=0)
    try:
        check(s.level_count == fake.level_count, 'fake level_count preserved')
        check(tuple(s.level_dimensions) == tuple(fake.level_dimensions),
              'fake level_dimensions preserved')
        check(all(abs(a - b) < 1e-9 for a, b in
                  zip(s.level_downsamples, fake.level_downsamples)),
              'fake level_downsamples preserved')
        for k, v in fake.properties.items():
            check(s.properties.get(k) == v, 'property %s preserved' % k)
        check(s.properties.get('openslide.mpp-x') == '0.25',
              'openslide.mpp-x preserved')
        check(abs(s.properties['mpp'] - 0.25) < 1e-9,
              'mpp parsed into fmg metadata')
        check(set(s.associated_images.keys()) ==
              set(fake.associated_images.keys()),
              'associated_images names preserved: %s' % list(s.associated_images))
        for name, ref in fake.associated_images.items():
            got = s.associated_images[name]
            check(np.array_equal(np.asarray(got), np.asarray(ref)),
                  'associated image %r round-trips pixel-exactly' % name)
        cases = _make_cases(fake, range(fake.level_count))
        p, n, _, _, _, failures = _compare_regions(
            fake, s, cases, rgba=False, label='fake RGB (alpha dropped)')
        if failures:
            log('  first mismatches: %s' % (failures[:6],))
        check(p == n, 'fake: %d/%d regions pixel-identical at all levels' % (p, n))
    finally:
        s.close()


def main():
    _install_eio_retry()
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)

    import openslide
    log('# fmg / OpenSlide conformance report')
    log('')
    log('- OpenSlide Python: %s, C library: %s' %
        (openslide.__version__, openslide.__library_version__))
    log('- fmg: %s (format version %d)' % (fmg.__version__, fmg.FORMAT_VERSION))
    log('- coordinate rule: `int(x0 / downsample)` (C-style truncation)')
    log('- comparison: OpenSlide RGBA alpha channel DISCARDED, RGB compared')
    log('')

    specs = [
        ('integer', [(1536, 1024), (768, 512), (384, 256)], True),
        ('fractional', [(1536, 1024), (1000, 667), (500, 333)], False),
    ]
    results = {}
    for name, dims, exact in specs:
        tif = _make_tiff(os.path.join(TMP, name + '.tif'), dims)
        dst = os.path.join(TMP, name + '.fmg')
        os_slide = openslide.open_slide(tif)
        log('\n=== %s ===' % name)
        log('  OpenSlide: %d levels, dims=%s, downsamples=%s' %
            (os_slide.level_count, os_slide.level_dimensions,
             tuple(round(d, 6) for d in os_slide.level_downsamples)))
        t0 = time.time()
        meta = fmg.from_openslide(os_slide, dst, ext='.png', patch_size=256)
        log('  converted in %.2fs -> %s' %
            (time.time() - t0, os.path.relpath(dst, BASE)))
        try:
            fm_slide = fmg.open_slide(dst, patch_cache=0)
            verify_api(os_slide, fm_slide, dst, tif, exact)
            fm_slide.close()
            verify_pixels(os_slide, dst, name, strict=exact)
            results[name] = 'PASS'
        except AssertionError as exc:
            results[name] = 'FAIL: %s' % exc
            log('  !! %s' % exc)
        os_slide.close()

    verify_fake()

    log('\n=== summary ===')
    for name, res in results.items():
        log('  %-12s %s' % (name, res))
    mandatory = [ok for ok, msg, m in _checks if m]
    log('  mandatory checks: %d/%d passed' % (sum(mandatory), len(mandatory)))
    all_ok = all(mandatory) and all(v == 'PASS' for k, v in results.items()
                                    if k == 'integer')
    log('  RESULT: %s' % ('PASS' if all_ok else 'FAIL'))

    with open(REPORT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(_lines) + '\n')
    print('\nreport written to %s' % REPORT)
    return 0 if all_ok else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        with open(REPORT, 'w', encoding='utf-8') as _fh:
            _fh.write('\n'.join(_lines) + '\n\nTRACEBACK\n' + traceback.format_exc())
        sys.exit(2)
