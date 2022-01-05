import numpy as np
import paddle

from models.yolo import myYOLO

from reprod_log import ReprodLogger

if __name__ == "__main__":
    # load model
    # the model is save into ~/YOLO_reprod/weights_trans/yolo_paddle.pdparams
    device = paddle.device.set_device("cpu")
    model = myYOLO(device=device, input_size=[416, 416], trainable=False)
    model.load_dict(paddle.load("../../weights_trans/yolo_paddle.pdparams"))
    model.eval()

    # def logger
    reprod_logger = ReprodLogger()

    # read fake data
    fake_data = np.load("../../fake_data/fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)

    # forward
    out = model(fake_data)
    out_np = np.concatenate((np.array(out[0]),np.array(out[1]).reshape([13,1]),np.array(out[2]).reshape([13,1])), axis=1)

    # save output 
    reprod_logger.add("logits", out_np)
    reprod_logger.save("forward_paddle.npy")

