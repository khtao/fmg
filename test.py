import fmg
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

class SlideData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.slide_image = fmg.open_slide(data_path)
        self.proporties = self.slide_image.properties
        print(self.proporties)
        self.read_size = 512
        # self.read_size = int((0.249427 * 512) / self.proporties['mpp'])
        # self.read_step = int(self.read_size * 0.8)
        # self.out_step = int(self.read_step * self.out_size / self.read_size)
        # self.start_xy, self.end_xy = self.get_image_region()
        self.weight, self.height = self.slide_image.dimensions
        self.len_x = int(self.weight / self.read_size)
        self.len_y = int(self.height / self.read_size)
        # self.is_open = False

    def __len__(self):
        return self.len_y * self.len_x

    def readroi(self, x, y, w, h):
        image = self.slide_image.read_region(x, y, w, h, 0)
        return image

    def __getitem__(self, item):
        # if not self.is_open:
        #     self.slide_image.open(self.data_path)
        #     self.is_open = True
        position_x = int(item % self.len_x) * self.read_size
        position_y = int(item // self.len_x) * self.read_size
        image = self.slide_image.read_region((position_x, position_y), 0, (self.read_size, self.read_size))
        position = np.array([position_x, position_y])
        image = np.array(image.convert('RGB'))
        return image, position


if __name__ == '__main__':
    pp = '/home/khtao/workplace/seafile/1159869.fmg'
    dataset = SlideData(pp)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=4)
    for kk, jj in tqdm(dataloader):
        pass
        # print(jj)

