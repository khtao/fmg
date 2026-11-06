# -*- coding: utf-8 -*-
"""fmg -- Fast Memory Graphics for big images, used like OpenSlide."""
from .__version__ import __version__
from .core import (
    AbstractSlide,
    CorruptTileError,
    FastMemoryGraphics,
    FmgError,
    FORMAT_VERSION,
    configure,
)

__all__ = [
    'open_slide', 'create_slide', 'FastMemoryGraphics', 'AbstractSlide',
    'FmgError', 'CorruptTileError', 'configure', 'FORMAT_VERSION',
    '__version__',
]


def open_slide(filename, **kwargs):
    """Open an existing .fmg slide for reading (lazy, fork-safe)."""
    return FastMemoryGraphics(filename, mode='r', **kwargs)


def create_slide(filename, **kwargs):
    """Create a writer for a new .fmg slide."""
    return FastMemoryGraphics(filename, mode='w', **kwargs)
