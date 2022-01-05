import numpy as np
import cv2
import paddle

from models.yolo import myYOLO
from data import *
import time

def base_transform(image, size, mean):
    x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

def vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names):
    for i, box in enumerate(bboxes):
        # print(scores)
        cls_indx = cls_inds[i]
        xmin, ymin, xmax, ymax = box
        # xmin = int(xmin) / 416 * h
        # ymin = int(ymin) / 416 * w
        # xmax = int(xmax) / 416 * h
        # ymax = int(ymax) / 416 * w       
        if scores[i] > thresh:
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)])
            print(int(cls_indx))
            print(class_colors[int(cls_indx)])
            # cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
            mess = '%s' % (class_names[int(cls_indx)])
            print(mess)
            img = cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return img


if __name__ == "__main__":
    # read predict image
    root = "../../markdown_img/000001.jpg"
    save = "../../markdown_img/000001_result.jpg"
    img = cv2.imread(root)
    # img = cv2.resize(img, (416, 416),interpolation = cv2.INTER_AREA)
    h, w, _ = img.shape
    print(h, w)
    data = base_transform(img, ([416, 416]), (0, 0, 0))
    data = paddle.to_tensor(data)
    data = paddle.transpose(data, perm=[2,0,1])
    data = paddle.reshape(data, [1, 3, 416, 416])

    # load model
    device = paddle.device.set_device("cpu")
    model = myYOLO(device=device, input_size=[416, 416], trainable=False)
    model.load_dict(paddle.load("./weights/voc/yolo/yolo_best_model.pdparams"))
    model.eval()
    print('Finished loading model!')

    # forward
    t0 = time.time()
    bboxes, scores, cls_inds = model(data)
    print("detection time used ", time.time() - t0, "s")

    # show detection image
    scale = np.array([[w, h, w, h]])
    # map the boxes to origin image scale
    bboxes *= scale
    num_classes = 20
    COLORS = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(num_classes)]
    print(COLORS)
    img_processed = vis(img, bboxes, scores, cls_inds, thresh=0, class_colors=COLORS, class_names=VOC_CLASSES)
    cv2.imwrite(save, img_processed)
    print('successed!')
