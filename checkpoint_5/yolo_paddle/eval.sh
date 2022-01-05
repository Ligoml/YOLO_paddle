export PYTHONPATH=.

python eval.py  \
    --dataset_dir /home/bml/.storage/mnt/v-uimvattx3skxxwr7/org/workflow/VOCdevkit/ \
    --trained_model ./weights/voc/yolo/yolo_best_model.pdparams \
    --gpu