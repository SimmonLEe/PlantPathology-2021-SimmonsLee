# PlantPathology-2021
Description:Kaggle_Plant_Pathology (CV Multi_Label_Classification)<br/>
Device: 1X2080
# Method:
1.DataAugmentation:{


    1.Resize
    2.RandomCrop
    3.RandomVerticalFlip(0.5)
    4.RandomHorizontalFlip(0.5)
    5.RandomRoatation
    6.Normalize


   }<br/>
   
2.Model:{

    Resnet50
}

3.Training{

    1.optimizer:Adam
    2.schedule: CosineAnnealingWarmRestarts
    3.Init_lr : 0.01
    4.epoch : 200
    5.batchsize : 64
    6.loss_function : BCEWithLogitsLoss
  }
	
4.trick{

    1.label_smoothing,
    2.Image Pyramid,
    
 }
	
# Score:

# Here is my way to label it:

