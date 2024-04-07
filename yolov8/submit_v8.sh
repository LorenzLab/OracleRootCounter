#!/bin/bash

#SBATCH -A ACD112211
#SBATCH -N 1
#SBATCH --mem=360G
#SBATCH -c 16
#SBATCH -p gp4d
#SBATCH --gres=gpu:4

module load cuda/11.7
module load miniconda3

conda activate py310-torch2.2.2

srun yolo train model=yolov8n.pt data=ORACLE_DS.yaml epochs=1000 batch=440 imgsz=500 workers=16 device=0,1,2,3
# srun python -X faulthandler train_dual.py --workers 8 --device 0 --batch 108 --data ORACLE_DS.yaml --img 500 --cfg models/detect/yolov9-s.yaml --weights '' --name yolov9-s --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

