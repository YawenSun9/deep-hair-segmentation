CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet --lr 0.0001 --workers 1 --epochs 25 --batch-size 32 --gpu-ids 0 --checkname deeplab-resnet-testdataargu --eval-interval 1 --dataset lfw
