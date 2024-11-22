# train yolov9 models structure prune

python struc_prune_train_dual.py --workers 8 --device 0 --batch 16 --data ../data/mini_project/data.yaml --img 640 --nosave --cfg models/detect/yolov9-c.yaml --weights '' --name struc-prune-yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 15 >> ./result/yolov9_structure_prune.txt