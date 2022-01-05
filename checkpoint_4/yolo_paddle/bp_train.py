import numpy as np
import paddle
import paddle.nn as nn
from models.yolo import myYOLO
import tools

from reprod_log import ReprodLogger


if __name__ == "__main__":
    # def logger
    reprod_logger = ReprodLogger()

    # load model
    # the model is save into ~/YOLO_reprod/weights_trans/yolo_paddle.pdparams
    train_size = [416, 416]
    # base_lr = 1e-4
    # momentum = 0.9
    # weight_decay = 5e-4
    base_lr = 0.0001
    momentum = 0.9
    weight_decay = 0.0
    device = paddle.device.set_device("gpu")

    model = myYOLO(device=device, input_size=[416, 416], trainable=True)
    model.load_dict(paddle.load("../../weights_trans/yolo_paddle.pdparams"))
    model.train()

    # read or gen fake data
    fake_data = np.load("../../fake_data/fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)

    fake_label = np.load("../../fake_data/fake_label.npy")
    fake_label = fake_label.tolist()

    # optimizer setup
    tmp_lr = base_lr
    optimizer=paddle.optimizer.Momentum(parameters=model.parameters(), 
                                        learning_rate=base_lr, 
                                        momentum=momentum,
                                        weight_decay=weight_decay,
                                        use_nesterov=False)

    total_loss_list = []
    conf_loss_list = []
    cls_loss_list = []
    txtytwth_loss_list = []
    max_epoch = 5
    cos = True
    
    for epoch in range(max_epoch):
        model.set_grid(train_size)

        # make train label
        targets = [label for label in fake_label]
        targets = tools.gt_creator(input_size=train_size, stride=model.stride, label_lists=targets)
        targets = paddle.to_tensor(targets)
            
        # forward and loss
        conf_loss, cls_loss, txtytwth_loss, total_loss = model(fake_data, target=targets)

        # backprop
        total_loss.backward()        
        optimizer.step()
        optimizer.clear_grad()
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
    reprod_logger.save("bp_align_paddle.npy")