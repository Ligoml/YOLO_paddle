import numpy as np
import torch
import paddle

import warnings
warnings.filterwarnings("ignore")


def transfer():
    input_fp = "yolo_torch.pth"
    output_fp = "yolo_paddle.pdparams"
    torch_dict = torch.load(input_fp)
    paddle_dict = {}
    fc_names = [
        "classifier.1.weight", "classifier.4.weight", "classifier.6.weight"
    ]
    for key in torch_dict:
        weight = torch_dict[key].cpu().detach().numpy()
        flag = [i in key for i in fc_names]
        if any(flag):
            print("weight {} need to be trans".format(key))
            weight = weight.transpose()
        if key[-12:] == "running_mean":
            key = key[:-12] + '_mean'
        if key[-11:] == "running_var":
            key = key[:-11] + '_variance'
        paddle_dict[key] = weight
    paddle.save(paddle_dict, output_fp)


if __name__ == '__main__':
    transfer()