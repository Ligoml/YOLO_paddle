import numpy as np
import paddle
import math

from models.yolo import myYOLO

from reprod_log import ReprodLogger


if __name__ == "__main__":
    # def logger
    reprod_logger = ReprodLogger()

    base_lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4

    # load model
    # the model is save into ~/YOLO_reprod/weights_trans/yolo_paddle.pdparams
    device = paddle.device.set_device("cpu")
    model = myYOLO(device=device, input_size=[416, 416], trainable=False)
    model.load_dict(paddle.load("../../weights_trans/yolo_paddle.pdparams"))

    tmp_lr = base_lr
    optimizer=paddle.optimizer.Momentum(parameters=model.parameters(), 
                                        learning_rate=base_lr, 
                                        momentum=momentum,
                                        weight_decay=weight_decay,
                                        use_nesterov=False)

    model.eval()

    cos = True
    max_epoch = 90
    paddle_lr_list = []
    for epoch in range(max_epoch):
        # use cos lr
        if cos and epoch > 20 and epoch <= max_epoch - 20:
            # use cos lr
            tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1./ (max_epoch-20)))
            optimizer.set_lr(tmp_lr)

        elif cos and epoch > max_epoch - 20:
            tmp_lr = 0.00001
            optimizer.set_lr(tmp_lr)
        
        # # use step lr
        # else:
        #     if epoch in cfg['lr_epoch']:
        #         tmp_lr = tmp_lr * 0.1
        #         optimizer.set_lr(tmp_lr)
        paddle_lr_list.append(tmp_lr)
        optimizer.step()
        optimizer.clear_grad()

    # save output 
    reprod_logger.add("lr", np.array(paddle_lr_list))
    reprod_logger.save("lr_paddle.npy")