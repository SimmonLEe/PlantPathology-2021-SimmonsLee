import torch
from torch.utils.data import DataLoader
import time
from tensorboardX import SummaryWriter
import os

from predict import load_weight
from config import Config
from plant_datasets import MyData
# 可视化
writer = SummaryWriter("./logs/record/")

# 获取配置信息类
config = Config()

# 获取数据
train_db = MyData(config.Train_Images_Dir, csv_path=config.Train_Csv_File_path, transform=config.train_transforms)
train_db_loader = DataLoader(train_db, shuffle=True, batch_size=config.train_parameters["batch"], num_workers=40)

# 获取模型
model = config.model.to(config.Device)
# model = load_weight(config, model)
# 冻结所有层 除了最后一层
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.fc.parameters():
#     param.requires_grad = True
# 是否断点续训
start_epoch = 0


loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.train_parameters["lr"])
schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=0, T_0=config.train_parameters["epoch"])

# 表示量
best_loss = 0.05
# 开始训练
print("start training")
for epoch in range(start_epoch, config.train_parameters["epoch"]):
    model.train()
    # 70个epoch后开始训练所有的层
    # if epoch == 10:
    #     for param in model.parameters():
    #         param.requires_grad = True
    start_time = time.time()
    train_loss_total = 0
    file_time = time.time()
    IO_TIME = 0
    PROCESS_TIME = 0
    BACKWARD_TIME = 0
    for step, (x, y) in enumerate(train_db_loader):
        IO_TIME += (time.time() - file_time)
        y = y.type(torch.FloatTensor)
        if config.GPU:
            x = x.cuda()
            y = y.cuda()
        process_time = time.time()
        pred = model(x)
        train_loss = loss_function(pred, y)
        PROCESS_TIME += (time.time() - process_time)
        optimizer.zero_grad()
        backward_time = time.time()
        train_loss.backward()
        optimizer.step()
        BACKWARD_TIME += (time.time() - backward_time)
        train_loss_total += train_loss
        file_time = time.time()
    avg_loss = train_loss_total / (step + 1)
    writer.add_scalar("train_epoch_loss", avg_loss, global_step=epoch)
    writer.add_scalar("lr", schedule.get_last_lr()[0], epoch)
    print("\nepoch:{0} train_loss:{1:.5f} lr:{2:.5f} cost_time:{3:.5f}".format(epoch, avg_loss, schedule.get_last_lr()[0], time.time() - start_time))
    print("IO_TIME:{:.5f}  PROCESS_TIME:{:.5f} BACKWARD_TIME:{:.5f}".format(IO_TIME, PROCESS_TIME, BACKWARD_TIME))
    # 调整学习率
    schedule.step()
    # save checkpoint
    if avg_loss < best_loss:
        checkpoint = {
            "weight": model.state_dict(),
            "start_epoch": 0,
            "optimizer": optimizer,
            "schedule": schedule
        }
        best_loss = avg_loss
        save_dir_path = config.train_parameters["save_path"] + "{:.4f}".format(best_loss.item()) + ".pth"
        save_dir = config.train_parameters["save_path"]
        while not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(checkpoint, save_dir_path)



