import numpy as np
import torch
import torch.nn as nn
from models.yolo import myYOLO
import tools

from reprod_log import ReprodLogger


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    # def logger
    reprod_logger = ReprodLogger()

    # load model
    # the model is save into ~/YOLO_reprod/weights_trans/yolo_torch.pth
    train_size = [416, 416]
    # base_lr = 1e-4
    # momentum = 0.9
    # weight_decay = 5e-4
    base_lr = 0.0001
    momentum = 0.9
    weight_decay = 0.0
    device = torch.device("cuda")

    model = myYOLO(device=device, input_size=train_size, trainable=True)
    model.load_state_dict(torch.load("../../weights_trans/yolo_torch.pth"))
    model.to(device).train()

    # read or gen fake data
    fake_data = np.load("../../fake_data/fake_data.npy")
    fake_data = torch.from_numpy(fake_data)

    fake_label = np.load("../../fake_data/fake_label.npy")
    fake_label = fake_label.tolist()

    # optimizer setup
    tmp_lr = base_lr
    optimizer = torch.optim.SGD(model.parameters(), 
                            lr=base_lr, 
                            momentum=momentum,
                            weight_decay=weight_decay
                            )

    total_loss_list = []
    conf_loss_list = []
    cls_loss_list = []
    txtytwth_loss_list = []
    max_epoch = 5
    cos = True

    for epoch in range(max_epoch):
        # use cos lr
        # if cos and epoch > 20 and epoch <= max_epoch - 20:
        #     # use cos lr
        #     tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1./ (max_epoch-20)))
        #     set_lr(optimizer, tmp_lr)

        # elif cos and epoch > max_epoch - 20:
        #     tmp_lr = 0.00001
        #     set_lr(optimizer, tmp_lr)
        
        # # use step lr
        # else:
        #     if epoch in cfg['lr_epoch']:
        #         tmp_lr = tmp_lr * 0.1
        #         set_lr(optimizer, tmp_lr)
        
        # to device
        fake_data = fake_data.to(device)
        model.set_grid(train_size)

        # make train label
        targets = [label for label in fake_label]
        targets = tools.gt_creator(input_size=train_size, stride=model.stride, label_lists=targets)
        targets = torch.tensor(targets).float().to(device)
            
        # forward and loss
        conf_loss, cls_loss, txtytwth_loss, total_loss = model(fake_data, target=targets)

        # backprop
        total_loss.backward()        
        optimizer.step()
        optimizer.zero_grad()
        total_loss_list.append(total_loss)
        conf_loss_list.append(conf_loss)
        cls_loss_list.append(cls_loss)
        txtytwth_loss_list.append(txtytwth_loss)

    # save output 
    for idx, loss in enumerate(conf_loss_list):
        reprod_logger.add(f"conf_loss_{idx}", loss.detach().cpu().numpy())
    for idx, loss in enumerate(cls_loss_list):
        reprod_logger.add(f"cls_loss_{idx}", loss.detach().cpu().numpy())
    for idx, loss in enumerate(txtytwth_loss_list):
        reprod_logger.add(f"txtytwth_loss_{idx}", loss.detach().cpu().numpy())   
    for idx, loss in enumerate(total_loss_list):
        reprod_logger.add(f"total_loss_{idx}", loss.detach().cpu().numpy())
    reprod_logger.save("bp_align_torch.npy")