# -*- coding: utf-8 -*-
"""fmg -- Fast Memory Graphics for big images, used like OpenSlide.

The public API mirrors ``openslide`` so that code written for OpenSlide can be
pointed at an ``.fmg`` slide unchanged.  See ``fmg.from_openslide()`` to
convert an existing slide into the fmg format.
"""
from .__version__ import __version__
from .core import (
    AbstractSlide,
    CorruptTileError,
    FastMemoryGraphics,
    FmgError,
    FORMAT_VERSION,
    OpenSlideError,
    OpenSlideUnsupportedFormatError,
    PROPERTY_NAME_BACKGROUND_COLOR,
    PROPERTY_NAME_BARCODE,
    PROPERTY_NAME_BOUNDS_HEIGHT,
    PROPERTY_NAME_BOUNDS_WIDTH,
    PROPERTY_NAME_BOUNDS_X,
    PROPERTY_NAME_BOUNDS_Y,
    PROPERTY_NAME_COMMENT,
    PROPERTY_NAME_MPP_X,
    PROPERTY_NAME_MPP_Y,
    PROPERTY_NAME_OBJECTIVE_POWER,
    PROPERTY_NAME_QUICKHASH1,
    PROPERTY_NAME_VENDOR,
    configure,
    detect_format,
    from_openslide,
)

__all__ = [
    # OpenSlide-compatible surface
    'open_slide', 'detect_format', 'AbstractSlide', 'OpenSlideError',
    'OpenSlideUnsupportedFormatError', 'PROPERTY_NAME_BACKGROUND_COLOR',
    'PROPERTY_NAME_BARCODE', 'PROPERTY_NAME_BOUNDS_HEIGHT',
    'PROPERTY_NAME_BOUNDS_WIDTH', 'PROPERTY_NAME_BOUNDS_X',
    'PROPERTY_NAME_BOUNDS_Y', 'PROPERTY_NAME_COMMENT',
    'PROPERTY_NAME_MPP_X', 'PROPERTY_NAME_MPP_Y',
    'PROPERTY_NAME_OBJECTIVE_POWER', 'PROPERTY_NAME_QUICKHASH1',
    'PROPERTY_NAME_VENDOR',
    # fmg-specific surface
    'create_slide', 'from_openslide', 'FastMemoryGraphics', 'FmgError',
    'CorruptTileError', 'configure', 'FORMAT_VERSION', '__version__',
]


def open_slide(filename, **kwargs):
    """Open an existing .fmg slide for reading (lazy, fork-safe).

    ``rgba=True`` makes ``read_region`` return RGBA images like OpenSlide;
    the default is RGB, which is the fmg 0.4.x behaviour.
    """
    return FastMemoryGraphics(filename, mode='r', **kwargs)


def create_slide(filename, **kwargs):
    """Create a writer for a new .fmg slide."""
    return FastMemoryGraphics(filename, mode='w', **kwargs)
