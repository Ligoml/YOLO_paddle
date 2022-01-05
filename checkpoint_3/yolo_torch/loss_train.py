import numpy as np
import torch
import torch.nn as nn

from models.yolo import myYOLO
from utils.augmentations import SSDAugmentation
import tools

from reprod_log import ReprodLogger

if __name__ == "__main__":
    # def logger
    reprod_logger = ReprodLogger()

    # load model
    # the model is save into ~/YOLO_reprod/weights_trans/yolo_torch.pth
    device = torch.device("cpu")
    model = myYOLO(device=device, input_size=[416, 416], trainable=True)
    model.load_state_dict(torch.load("../../weights_trans/yolo_torch.pth"))
    model.eval()

    # read or gen fake data
    fake_data = np.load("../../fake_data/fake_data.npy")
    fake_data = torch.from_numpy(fake_data)

    fake_label = np.load("../../fake_data/fake_label.npy")
    fake_label = fake_label.tolist()

    # make train label
    targets = [label for label in fake_label]
    targets = tools.gt_creator(input_size=[416, 416], stride=model.stride, label_lists=targets)
    targets = torch.tensor(targets).float().to(device)
            
    # forward and loss
    conf_loss, cls_loss, txtytwth_loss, total_loss = model(fake_data, target=targets)

    # save output 
    reprod_logger.add("conf_loss", conf_loss.cpu().detach().numpy())
    reprod_logger.add("cls_loss", cls_loss.cpu().detach().numpy())
    reprod_logger.add("txtytwth_loss", txtytwth_loss.cpu().detach().numpy())
    reprod_logger.add("total_loss", total_loss.cpu().detach().numpy())
    reprod_logger.save("loss_torch.npy")