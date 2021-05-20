import torch
import torchvision
import pandas as pd
from config import Config


def create_model(config, pretrained=False):
    model = torchvision.models.resnet50(pretrained)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=model.fc.in_features,
                        out_features=config.Classes),
        torch.nn.Sigmoid()
    ).to(config.Device)
    return load_weight(model)


def load_weight(config, model):
    checkpoint = torch.load(config.model_weight_save_path, map_location=config.Device)
    model.load_state_dict(checkpoint["weight"])
    model.eval()
    print("load weight complete")
    return model


def submit(model, loader, config, classes):
    image_ids = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
    thresh = torch.tensor([0.6, 0.6, 0.5, 0.5, 0.5])
    for idx, (x, _) in enumerate(loader):
        if config.GPU:
            x = x.cuda()
        x = x.type(torch.FloatTensor)
        output = model(x).reshape(-1)
        if torch.max(output).item() < 0.25:
            pred = "healthy"
        else:
            l = []
            bool_output = output > thresh
            for i, t in enumerate(bool_output):
                if t.item():
                    l.append(classes[str(i)])
            pred = " ".join(l)
        image_ids.iloc[idx]["labels"] = pred

    image_ids.set_index('image', inplace=True)
    image_ids.to_csv(config.SUBMISSION_FILE)
    return image_ids


def main():
    config = Config()
    model = create_model(config)




if __name__ == "__main__":
    main()