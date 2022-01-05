export PYTHONPATH=.

python metric_eval.py  \
    --dataset_dir /home/bml/.storage/mnt/v-uimvattx3skxxwr7/org/workflow/VOCdevkit/ \
    --trained_model ../../weights_trans/yolo_torch.pth \
    --cuda