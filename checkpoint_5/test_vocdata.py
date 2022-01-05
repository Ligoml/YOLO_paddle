import os
import sys
import cv2
import numpy as np
import paddle
import torch
from reprod_log import ReprodLogger, ReprodDiffHelper

train_size = [416, 416]
val_size = [416, 416]

def base_transform(image, size, mean):
    x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

class BaseTransform:
        def __init__(self, size, mean):
            self.size = size
            self.mean = np.array(mean, dtype=np.float32)

        def __call__(self, image, boxes=None, labels=None):
            return base_transform(image, self.size, self.mean), boxes, labels

def paddle_detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(paddle.to_tensor(sample[1]))
    return paddle.stack(imgs, 0), targets


def torch_detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def build_paddle_data_pipeline():
    sys.path.insert(0, "./yolo_paddle/")
    from yolo_paddle.data.voc0712 import VOCDetection
    from yolo_paddle.utils.augmentations import SSDAugmentation

    data_dir = "/home/bml/.storage/mnt/v-uimvattx3skxxwr7/org/workflow/VOCdevkit/"
    num_classes = 20

    # initialize random seed
    paddle.seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    paddle_dataset = VOCDetection(root=data_dir, 
                            img_size=train_size[0],
                            transform=BaseTransform((train_size), (0, 0, 0))
                            )
    paddle_dataloader = paddle.io.DataLoader(
                    paddle_dataset, 
                    batch_size=32, 
                    shuffle=False, 
                    collate_fn=paddle_detection_collate,
                    num_workers=8,
                    drop_last=True
                    )
    sys.path.pop(0)

    return paddle_dataset, paddle_dataloader


def build_torch_data_pipeline():
    sys.path.insert(0, "./yolo_torch/")
    from yolo_torch.data.voc0712 import VOCDetection
    from yolo_torch.utils.augmentations import SSDAugmentation

    data_dir = "/home/bml/.storage/mnt/v-uimvattx3skxxwr7/org/workflow/VOCdevkit/"
    num_classes = 20

    # initialize random seed
    SEED = 111
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    torch_dataset = VOCDetection(root=data_dir, 
                            img_size=train_size[0],
                            transform=BaseTransform((train_size), (0, 0, 0))
                            )
    torch_dataloader = torch.utils.data.DataLoader(
                    torch_dataset, 
                    batch_size=32, 
                    shuffle=False, 
                    collate_fn=torch_detection_collate,
                    num_workers=8,
                    pin_memory=True
                    )
    sys.path.pop(0)

    return torch_dataset, torch_dataloader


def test_data_pipeline():
    diff_helper = ReprodDiffHelper()
    paddle_dataset, paddle_dataloader = build_paddle_data_pipeline()
    torch_dataset, torch_dataloader = build_torch_data_pipeline()

    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()

    logger_paddle_data.add("length", np.array(len(paddle_dataset)))
    logger_torch_data.add("length", np.array(len(torch_dataset)))

    # random choose 5 images and check
    for idx in range(5):
        rnd_idx = np.random.randint(0, len(paddle_dataset))
        logger_paddle_data.add(f"dataset_{idx}",
                               paddle_dataset[rnd_idx][0].numpy())
        logger_torch_data.add(f"dataset_{idx}",
                              torch_dataset[rnd_idx][0].detach().cpu().numpy())

    for idx, (paddle_batch, torch_batch
              ) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if idx >= 5:
            break
        logger_paddle_data.add(f"dataloader_{idx}", paddle_batch[0].numpy())
        logger_torch_data.add(f"dataloader_{idx}",
                              torch_batch[0].detach().cpu().numpy())

    diff_helper.compare_info(logger_paddle_data.data, logger_torch_data.data)
    diff_helper.report(path="data_diff.log")

if __name__ == "__main__":
    test_data_pipeline()