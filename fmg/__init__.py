from .core import FastMemoryGraphics


def open_slide(filename):
    return FastMemoryGraphics(filename)


def create_slide(filename):
    return FastMemoryGraphics(filename, 'w')
