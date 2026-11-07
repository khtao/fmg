#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package metadata for fmg.

Publishing is done with ``python -m build`` + ``twine upload``; see RELEASE.md.
There is intentionally no ``upload`` command here: ``python setup.py upload``
was removed from setuptools, and shelling out to ``twine upload dist/*`` from
inside a build command is fragile and easy to trigger by accident.

The version comes from fmg/__version__.py via exec(), not by importing the
package, so an isolated build never needs cv2 / lmdb / numpy / pillow present.
"""
import io
import os

from setuptools import find_packages, setup

NAME = 'fmg'
DESCRIPTION = 'Fast Memory Graphics for big image, use like OpenSlide'
URL = 'https://github.com/khtao/fmg'
EMAIL = 'khtao@live.cn'
AUTHOR = 'khtao'
REQUIRES_PYTHON = '>=3.7'

REQUIRED = ['lmdb>=1.0', 'opencv-python>=4.0', 'numpy>=1.17', 'pillow>=7.0']
EXTRAS = {
    # Only needed to convert from a filename with from_openslide(), and for
    # verify_openslide_compat.py.  fmg itself does not import openslide.
    'openslide': ['openslide-python>=1.2', 'openslide-bin>=4.0',
                  'tifffile>=2022'],
}

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

about = {}
with io.open(os.path.join(here, 'fmg', '__version__.py'), encoding='utf-8') as f:
    exec(f.read(), about)

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    maintainer=AUTHOR,
    maintainer_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    project_urls={
        'Source': URL,
        'Issues': URL + '/issues',
        'Optimization notes': URL + '/blob/main/OPTIMIZATION.md',
        'OpenSlide compatibility': URL + '/blob/main/OPENSLIDE_COMPAT.md',
        'Release guide': URL + '/blob/main/RELEASE.md',
    },
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*', 'tests.*']),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    license_files=('LICENSE',),
    keywords=[
        'whole slide image', 'WSI', 'digital pathology', 'LMDB',
        'OpenSlide', 'openslide-compatible', 'tiled image', 'microscopy',
        'remote sensing',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Image Processing',
        # License classifier is deprecated by modern setuptools; the SPDX
        # short identifier is declared via license='MIT' above.
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
