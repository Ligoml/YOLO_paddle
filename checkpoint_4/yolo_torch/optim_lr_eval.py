import numpy as np
import torch
import math

from models.yolo import myYOLO

from reprod_log import ReprodLogger

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    # def logger
    reprod_logger = ReprodLogger()

    base_lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4

    # load model
    # the model is save into ~/YOLO_reprod/weights_trans/yolo_torch.pth
    device = torch.device("cpu")
    model = myYOLO(device=device, input_size=[416, 416], trainable=False)
    model.load_state_dict(torch.load("../../weights_trans/yolo_torch.pth"))

    tmp_lr = base_lr
    optimizer = torch.optim.SGD(model.parameters(), 
                            lr=base_lr, 
                            momentum=momentum,
                            weight_decay=weight_decay
                            )
    model.eval()

    cos = True
    max_epoch = 90
    torch_lr_list = []
    for epoch in range(max_epoch):
        # use cos lr
        if cos and epoch > 20 and epoch <= max_epoch - 20:
            # use cos lr
            tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1./ (max_epoch-20)))
            set_lr(optimizer, tmp_lr)

        elif cos and epoch > max_epoch - 20:
            tmp_lr = 0.00001
            set_lr(optimizer, tmp_lr)
        
        # # use step lr
        # else:
        #     if epoch in cfg['lr_epoch']:
        #         tmp_lr = tmp_lr * 0.1
        #         set_lr(optimizer, tmp_lr)
        torch_lr_list.append(tmp_lr)
        optimizer.step()
        optimizer.zero_grad()

    # save output 
    reprod_logger.add("lr", np.array(torch_lr_list))
    reprod_logger.save("lr_torch.npy")