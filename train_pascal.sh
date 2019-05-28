CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet --lr 0.01 --workers 1 --epochs 4 --batch-size 16 --gpu-ids 0 --checkname deeplab-resnet-pascal --eval-interval 1 --dataset pascal
