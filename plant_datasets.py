from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import torch
from torch.nn.functional import one_hot


class MyData(Dataset):
    def __init__(self, images_dir, csv_path=None, txt_path=None, transform=None):
        super(MyData, self).__init__()
        # 获取csv_文件地址
        self.csv_path = csv_path
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError("请检查参数 csv_path 路径 | Please check the parameter csv_path")
        self.transform = transform
        self.images_dir = images_dir
        self.datasets = []
        self.Header = None
        self.get_classes()
        self.classes = {
            "scab": 0,
            "complex": 1,
            "rust": 2,
            "frog_eye_leaf_spot": 3,
            'powdery_mildew': 4
        }
        self.get_datas()


    # 获取文件对应的图片路径以及标签
    def get_classes(self):
        print("csv_path:{}".format(self.csv_path))
        data = pd.read_csv(self.csv_path)
        self.Header = [x for x in data]
        print("Headers:{}".format(self.Header))
    # 将图片与标签对应起来 用于__getitem__函数调用
    def get_datas(self):
        Data = pd.read_csv(self.csv_path)
        if 'image' in self.Header and 'labels' in self.Header:
            for x, y in zip(Data.image, Data.labels):
                target = torch.zeros(5)
                labels = y.split(" ")
                if "healthy" not in labels:
                    label = [self.classes[str(m)] for m in labels]
                    target[label] = 1.0
                image_path = self.images_dir + "/" + x
                target = self.label_smooth(target)
                self.datasets.append([image_path, target])

    def label_smooth(self, target, epsilon=0.05):
        target = target * (1 - epsilon) + epsilon / target.size(0)
        return target

    def __getitem__(self, item):
        img_path, label = self.datasets[item][0], self.datasets[item][1]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.datasets)