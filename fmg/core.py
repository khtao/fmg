import json
import math
import os
import shutil

import cv2
import lmdb
import numpy as np
from PIL import Image


class AbstractSlide:
    """The base class of a slide object."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Close the slide."""
        raise NotImplementedError

    @property
    def level_count(self):
        """The number of levels in the image."""
        raise NotImplementedError

    @property
    def level_dimensions(self):
        """A list of (width, height) tuples, one for each level of the image.

        level_dimensions[n] contains the dimensions of level n."""
        raise NotImplementedError

    @property
    def dimensions(self):
        """A (width, height) tuple for level 0 of the image."""
        return self.level_dimensions[0]

    @property
    def level_downsamples(self):
        """A list of downsampling factors for each level of the image.

        level_downsample[n] contains the downsample factor of level n."""
        raise NotImplementedError

    @property
    def properties(self):
        """Metadata about the image.

        This is a map: property name -> property value."""
        raise NotImplementedError

    def read_region(self, location, level, size):
        """Return a PIL.Image containing the contents of the region.

        location: (x, y) tuple giving the top left pixel in the level 0
                  reference frame.
        level:    the level number.
        size:     (width, height) tuple giving the region size."""
        raise NotImplementedError

    def save_from_numpy(self, data, mpp, ext, params):
        raise NotImplementedError

    def save_from_tiles(self, patches, locations, level, ext, params):
        raise NotImplementedError


class FastMemoryGraphics(AbstractSlide):
    """An open whole-slide image.

    close() is called automatically when the object is deleted.
    The object may be used as a context manager, in which case it will be
    closed upon exiting the context.

    If an operation fails, OpenSlideError is raised.  Note that OpenSlide
    has latching error semantics: once OpenSlideError is raised, all future
    operations on the OpenSlide object, other than close(), will fail.
    """

    def __init__(self, filename, mode='r'):
        """Open a whole-slide image."""
        AbstractSlide.__init__(self)
        self._filename = filename
        self._mode = mode
        if mode == 'r':
            self._osr = json.load(open(filename))
            self._data = [lmdb.open(os.path.join(self._filename[:-4], 'level' + str(i))).begin()
                          for i in range(self.level_count)]
            # self.check_data()
        elif mode == 'w':
            pass

    # def check_data(self):
    #     for i in range(self.level_count):
    #         txn = self._data[i].begin()
    #         dimension = txn.get(key='dimension'.encode())
    #         if dimension is not None:
    #             dimension_w = dimension.decode()
    #             dimension_r = str(self.level_dimensions[i])
    #             if dimension_r != dimension_w:
    #                 print('warning: read size %s does not match write size %s' % (dimension_w, dimension_r))

    def __repr__(self):
        return f'{self.__class__.__name__}({self._filename!r})'

    def close(self):
        """Close the OpenSlide object."""
        for dd in self._data:
            dd.close()

    @property
    def level_count(self):
        """The number of levels in the image."""
        return self._osr['level_count']

    @property
    def level_dimensions(self):
        """A list of (width, height) tuples, one for each level of the image.

        level_dimensions[n] contains the dimensions of level n."""
        return self._osr['level_dimensions']

    @property
    def level_downsamples(self):
        """A list of downsampling factors for each level of the image.

        level_downsample[n] contains the downsample factor of level n."""
        return self._osr['level_downsamples']

    @property
    def properties(self):
        """Metadata about the image.

        This is a map: property name -> property value."""
        return self._osr

    def read_region(self, location, level, size):

        """Return a PIL.Image containing the contents of the region.

        location: (x, y) tuple giving the top left pixel in the level 0
                  reference frame.
        level:    the level number.
        size:     (width, height) tuple giving the region size.

        Unlike in the C interface, the image data returned by this
        function is not premultiplied."""
        assert self._mode == 'r', 'must be in read mode!'
        x_start, y_start = location
        x_end, y_end = x_start + size[0], y_start + size[1]
        dimension = self.level_dimensions[level]
        step = self._osr['patch_size']
        x_min = math.floor(x_start / step) * step
        y_min = math.floor(y_start / step) * step
        x_max = math.ceil(x_end / step) * step
        y_max = math.ceil(y_end / step) * step
        image = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
        txn = self._data[level]
        for w in range(x_min, min(x_max, dimension[0]), step):
            for h in range(y_min, min(y_max, dimension[1]), step):
                w_max = min(w + step, dimension[0])
                h_max = min(h + step, dimension[1])
                buff = txn.get(key=str((w, h, w_max, h_max)).encode())
                buff = np.frombuffer(buff, dtype=np.uint8)
                buff = cv2.imdecode(buff, cv2.IMREAD_COLOR)
                image[h - y_min:h_max - y_min, w - x_min:w_max - x_min, :] = buff
        image = image[y_start - y_min:y_end - y_min, x_start - x_min:x_end - x_min, :]
        return Image.fromarray(image)

    def save_from_numpy(self, data, ext='.jpg',
                        params=[cv2.IMWRITE_JPEG_QUALITY, 90],
                        pyramid=False, mpp=None, patch_size=256):
        assert self._mode == 'w'
        w, h, c = data.shape
        level_count = 1
        level_dimensions = [[h, w]]
        level_downsamples = [[1.0, 1.0]]
        level_data = [data]
        if pyramid:
            level_dimension = [h, w]
            while True:
                if 256 <= level_dimension[0] <= 512 or 256 <= level_dimension[1] <= 512:
                    break
                else:
                    level_dimension[0] = level_dimension[0] // 2
                    level_dimension[1] = level_dimension[1] // 2
                    level_count += 1
                    level_downsamples.append([h / level_dimension[0], w / level_dimension[1]])
                    level_dimensions.append([level_dimension[0], level_dimension[1]])
                    level_data.append(level_data[-1][::2, ::2, :])

        properties = {}
        properties['level_count'] = level_count
        properties['level_dimensions'] = level_dimensions
        properties['dimension'] = level_dimensions[0]
        properties['level_downsamples'] = level_downsamples
        properties['mpp'] = float(mpp)
        properties['ext'] = ext
        properties['params'] = params
        properties['patch_size'] = patch_size
        self._osr = properties
        if os.path.exists(self._filename[:-4]):
            shutil.rmtree(self._filename[:-4])
        os.makedirs(self._filename[:-4])
        self._data = []
        for i in range(level_count):
            self._data.append(lmdb.open(os.path.join(self._filename[:-4], 'level' + str(i)), map_size=1099511627776))
        for i, dimension in enumerate(level_dimensions):
            txn = self._data[i].begin(write=True)
            txn.put(key='patch_size'.encode(), value=str(patch_size).encode())
            txn.put(key='mpp'.encode(), value=str(mpp).encode())
            txn.put(key='dimension'.encode(), value=str(dimension).encode())
            txn.commit()
            for x in range(0, dimension[0], patch_size):
                patches = []
                locations = []
                for y in range(0, dimension[1], patch_size):
                    x_max = min(x + patch_size, dimension[0])
                    y_max = min(y + patch_size, dimension[1])
                    patches.append(level_data[i][y:y_max, x:x_max, :])
                    locations.append((x, y, x_max, y_max))
                self.save_from_tiles(patches, locations, i, ext, params)
        fs = open(self._filename, 'w')
        fs.write(json.dumps(properties))
        fs.close()

    def save_from_tiles(self, patches, locations, level, ext, params):
        txn = self._data[level].begin(write=True)
        for p, l in zip(patches, locations):
            buff = cv2.imencode(ext, np.array(p, dtype=np.uint8), params)[1]
            txn.put(key=str(l).encode(), value=buff.tobytes())
        txn.commit()
