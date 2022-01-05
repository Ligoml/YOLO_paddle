import numpy as np
import torch

from models.yolo import myYOLO

from reprod_log import ReprodLogger

if __name__ == "__main__":
    # def logger
    reprod_logger = ReprodLogger()

    # load model
    # the model is save into ~/YOLO_reprod/weights_trans/yolo_torch.pth
    device = torch.device("cpu")
    model = myYOLO(device=device, input_size=[416, 416], trainable=False)
    model.load_state_dict(torch.load("../../weights_trans/yolo_torch.pth"))
    model.eval()

    # read fake data
    fake_data = np.load("../../fake_data/fake_data.npy")
    fake_data = torch.from_numpy(fake_data)

    # forward
    out = model(fake_data)
    out_np = np.concatenate((np.array(out[0]),np.array(out[1]).reshape([13,1]),np.array(out[2]).reshape([13,1])), axis=1)

    # save output 
    reprod_logger.add("logits", out_np)
    reprod_logger.save("forward_torch.npy")

