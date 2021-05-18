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
        self.SAMPLE_SUBMISSION_FILE = ""
        self.model_weight_save_path = ""

        self.GPU = True if torch.cuda.is_available() else False
        self.Device = torch.device("cuda:0") if self.GPU else torch.device("cpu")
        print("run on {}".format(self.Device))

        self.model = torchvision.models.resnet50(pretrained=True).to(self.Device)
        self.Classes = 5

        # modify last layer
        self.model.fc = nn.Sequential(
            nn.Linear(
                in_features=self.model.fc.in_features,
                out_features=self.Classes
            ),
        )

        self.scale = 1.5
        # self.Pre_Trim_Image_Height_Size = [200, 400, 600, 800]
        self.Pre_Trim_Image_Height_Size = 400
        # self.Pre_Trim_Image_Width_Size = [int(1.5 * x) for x in self.Pre_Trim_Image_Height_Size]
        self.Pre_Trim_Image_Width_Size = int(self.Pre_Trim_Image_Height_Size * self.scale)

        # self.Input_Image_Height_Size = [150, 300, 450, 600]
        self.Input_Image_Height_Size = 300
        # self.Input_Image_Width_Size = [int(1.5 * x) for x in self.Input_Image_Height_Size]
        self.Input_Image_Width_Size = int(self.Input_Image_Height_Size * self.scale)


        self.train_parameters = {
            "epoch": 300,
            "batch": 16,
            "lr": 0.01,
            "save_path": "./checkpoints/",
            "RESUME": False,
        }

        self.train_transforms = transforms.Compose([
            transforms.Resize((self.Pre_Trim_Image_Height_Size, self.Pre_Trim_Image_Width_Size)),

            transforms.RandomCrop((self.Input_Image_Height_Size, self.Input_Image_Width_Size)),

            # transforms.ColorJitter(),

            transforms.RandomVerticalFlip(0.5),

            transforms.RandomHorizontalFlip(0.5),

            transforms.RandomRotation(45),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),


        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((self.Input_Image_Height_Size, self.Input_Image_Width_Size)),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
