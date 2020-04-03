CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py hmdb51 RGB \
     --arch tea50 --num_segments 16 --gpus 0 1 2 3 4 5 6 7 \
     --gd 20 --lr 0.001 --lr_steps 10 20 -epochs 25 \
     --batch-size 64 -j 16 -dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres \
     --experiment_name=HMDB \
     --tune_from='../checkpoint/TEA_something_RGB_tea50_shift8_blockres_avg_segment16_e50/ckpt.best.pth.tar'
