from __future__ import division
import os
import random
import argparse
import time
import math
import numpy as np
import logging

import paddle
import paddle.optimizer as optim
paddle.disable_static()
from data import *
import tools
from utils.augmentations import SSDAugmentation
from utils.vocapi_evaluator import VOCAPIEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-v', '--version', default='yolo',
                        help='yolo')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('--dataset_dir', default='./datasets/VOCdevkit/',
                    help='Please input the dataset dir:') 
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')  
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('--max_epoch', type=int, default=160,
                        help='set max train epoch')                       
    parser.add_argument('-r', '--resume', default=None, type=str,help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='use gpu.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')

    return parser.parse_args()


def train():
    args = parse_args()

    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # use hi-res backbone
    if args.high_resolution:
        logging.info('use hi-res backbone')
        hr = True
    else:
        hr = False
    
    if args.gpu:
        logging.info('use gpu')
        device = paddle.device.set_device("gpu")
    else:
        device = paddle.device.set_device("cpu")

    # multi-scale
    if args.multi_scale:
        logging.info('use the multi-scale trick ...')
        train_size = [640, 640]
        val_size = [416, 416]
    else:
        train_size = [416, 416]
        val_size = [416, 416]

    # dataset and evaluator
    logging.info("Setting Arguments.. : %s" % args)
    logging.info("----------------------------------------------------------")
    logging.info('Loading the dataset...')

    if args.dataset == 'voc':
        # data_dir = VOC_ROOT
        data_dir = args.dataset_dir
        num_classes = 20
        dataset = VOCDetection(root=data_dir, 
                                img_size=train_size[0],
                                transform=SSDAugmentation(train_size)
                                )

        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=VOC_CLASSES
                                    )
    else:
        logging.warning('unknow dataset !! Only support voc and coco !!')
        exit(0)
    
    logging.info('Training model on: %s' % dataset.name)
    logging.info('The dataset size: %d' % len(dataset))
    logging.info("----------------------------------------------------------")

    # dataloader
    dataloader = paddle.io.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,
                    drop_last=True
                    )

    # build model
    if args.version == 'yolo':
        from models.yolo import myYOLO
        yolo_net = myYOLO(device, input_size=train_size, num_classes=num_classes, trainable=True)
        logging.info('Let us train yolo on the %s dataset ......' % (args.dataset))

    else:
        logging.warning('We only support YOLO !!!')
        exit()


    model = yolo_net
    model.train()

    # keep training
    if args.resume is not None:
        logging.info('keep training model: %s' % (args.resume))
        model.load_dict(paddle.load(args.resume))

    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer=paddle.optimizer.Momentum(parameters=model.parameters(), 
                                        learning_rate=args.lr, 
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        use_nesterov=True
                                        )
  
    max_epoch = args.max_epoch
    epoch_size = len(dataset) // args.batch_size

    # start training loop
    t0 = time.time()
    map_max = 0
    map_eval = 0
    map_max_model = 0
    lr_epoch = (60, 90, 160)

    for epoch in range(args.start_epoch, max_epoch):
        # use cos lr
        if args.cos and epoch > 20 and epoch <= max_epoch - 20:
            # use cos lr
            tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1./ (max_epoch-20)))
            optimizer.set_lr(tmp_lr)

        elif args.cos and epoch > max_epoch - 20:
            tmp_lr = 0.00001
            optimizer.set_lr(tmp_lr)
        
        # use step lr
        else:
            if epoch in lr_epoch:
                tmp_lr = tmp_lr * 0.1
                optimizer.set_lr(tmp_lr)
    

        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
                    optimizer.set_lr(tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    optimizer.set_lr(tmp_lr)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                size = random.randint(10, 19) * 32
                train_size = [size, size]
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = paddle.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            # make train label
            targets = [label.tolist() for label in targets]
            targets = tools.gt_creator(input_size=train_size, stride=yolo_net.stride, label_lists=targets)
            targets = paddle.to_tensor(targets)
            
            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets)

            # backprop
            total_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            # display
            if iter_i % 10 == 0:
                logging.info('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d ]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), train_size[0]))
                

            # evaluation
            if iter_i %500==0 and iter_i!=0:
                model.trainable = False
                model.set_grid(val_size)
                model.eval()

                # evaluate
                map_eval = evaluator.evaluate(model)
                if map_eval > map_max: 
                    map_max = map_eval
                    map_max_model = epoch + 1
                    paddle.save(model.state_dict(), os.path.join(path_to_save, 'yolo_best_model.pdparams'))
                # convert to training mode.
                model.trainable = True
                model.set_grid(train_size)
                model.train()

                # save model
                if (epoch + 1) % 10 == 0:
                    logging.info('Saving state, epoch: %d' % (epoch + 1)) 
                    paddle.save(model.state_dict(), os.path.join(path_to_save, 
                        args.version + '_' + repr(epoch + 1) + '.pdparams'))
            
    # logging best model
    logging.info('The best model is: yolo_%d.pdparams' % map_max_model)
    logging.info('The best eval map is: %2f'% map_max)


if __name__ == '__main__':
    logging.basicConfig(filename='./TrainLog.log', filemode='a', level=logging.INFO)
    train()
