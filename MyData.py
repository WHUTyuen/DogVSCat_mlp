from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np

class MyDatasets(Dataset):

    # 设计模式 代理模式  load-delay
    def __init__(self,imgpath):
        self.path = imgpath
        self.datapaths = os.listdir(imgpath)

    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, index):
        target = self.datapaths[index][0:1]
        imgname = self.datapaths[index]
        img = Image.open(os.path.join(self.path,imgname))
        data = np.array(img)
        data = torch.Tensor(data) / 255 - 0.5 #归一化去均值
        target = torch.tensor([int(target)])
        return data,target




if __name__ == '__main__':

    obj = MyDatasets("img")
    x = obj[0][0]
    y = obj[0][1]
    print(obj[0])
    x = x.numpy()
    x = (x + 0.5) * 255
    print(x)
    x = np.array(x,dtype=np.int8)
    # img = Image.fromarray(x)
    # img.show()
    print(x)
    img = Image.fromarray(x,"RGB")
    img.show()

