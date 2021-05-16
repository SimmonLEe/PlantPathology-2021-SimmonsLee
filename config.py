from torchvision import transforms
import torchvision
import torch
import torch.nn as nn
# 参数信息

class Config():
    def __init__(self):
        self.Root = "./"
        self.Train_Csv_File_path = self.Root + "planet/train.csv"
        self.Train_Images_Dir = self.Root + "planet/train_images/"
        self.Test_Images_Dir = self. Root + "planet/test_images/"

        self.GPU = True if torch.cuda.is_available() else False
        self.Device = torch.device("cuda:0") if self.GPU else torch.device("cup")
        print("run on {}".format(self.Device))

        self.model = torchvision.models.resnet50(pretrained=True).to(self.Device)
        self.Classes = 6

        # modify last layer
        self.model.fc = nn.Sequential(
            nn.Linear(
                in_features=self.model.fc.in_features,
                out_features=self.Classes
            ),
        )

        self.Pre_Trim_Image_size = 400
        self.Input_Image_Size = 224


        self.train_parameters = {
            "epoch": 200,
            "batch": 64,
            "lr": 0.001,
            "save_path": "",
            "RESUME": False,
        }

        self.train_transforms = transforms.Compose([
            transforms.Resize(self.Pre_Trim_Image_size),

            transforms.RandomCrop(self.Input_Image_Size),

            # transforms.ColorJitter(),

            transforms.RandomVerticalFlip(0.5),

            transforms.RandomHorizontalFlip(0.5),

            transforms.RandomRotation(45),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),


        ])
