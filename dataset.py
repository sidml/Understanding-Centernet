from torch.utils.data import Dataset
import cv2
import numpy as np


class ToyDataset(Dataset):
    """Car dataset."""

    def __init__(self, img_shape=(256,256),max_radius=64,
                    num_classes=1,max_objects=5):
        super().__init__()
        self.img_shape = np.array(img_shape)
        self.num_classes = num_classes
        self.max_width = 64
        self.max_height = 64
        self.max_radius = min(img_shape)//4
        self.max_objects = max_objects


        w, h = self.img_shape//4
        # prepare mesh center points
        x_arr = np.arange(w) + 0.5
        y_arr = np.arange(h) + 0.5
        self.xy_mesh = np.stack(np.meshgrid(x_arr, y_arr))  # [2, h, w]

    def __len__(self):
        return 1000

    def __getitem__(self, idx):

        im = np.zeros(self.img_shape,dtype=np.float32)
        heatmap = np.zeros((self.num_classes+4,self.img_shape[0]//4,self.img_shape[1]//4),dtype=np.float32)
        for _ in range(np.random.randint(0,5)):
            x,y = np.random.randint(0,self.img_shape[0]),np.random.randint(0,self.img_shape[1])
            radius = np.random.randint(10,self.max_radius)
            im = np.maximum(im,cv2.circle(im,(y,x),radius=radius,color=1,thickness=-1))

            center = np.array([x,y])/4
            x, y = np.floor(center).astype(np.int)
            # print('center,wh',center,wh)

            # sigma = gaussian_radius(wh)
            # dist_squared = np.sum((self.xy_mesh - center[:, None, None]) ** 2, axis=0)
            # gauss = np.exp(-1 * dist_squared / (2 * sigma ** 2))
            # heatmap[0, :, :] = np.maximum(heatmap[0, :, :], gauss)

            heatmap[0,x,y] = 1

            # size
            heatmap[-4:-2,x,y] = np.array([2*radius,2*radius])

            # offset
            heatmap[-2:, x,y] = center - np.floor(center)

        return im[None,:,:], heatmap