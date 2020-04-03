# test TEA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_models.py something \
    --arch='tea50' \
    --weight='./checkpoint/TEA_something_RGB_tea50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar' \
    --worker=6 --gpus 0 1 2 3 4 5 6 7 \
    --test_segments=8 --test_crops=3 \
    --batch_size=64 --shift --multi_clip_test \
    --clip_index=0 --full_res

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_models.py something \
    --arch='tea50' \
    --weight='./checkpoint/TEA_something_RGB_tea50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar' \
    --worker=6 --gpus 0 1 2 3 4 5 6 7 \
    --test_segments=8 --test_crops=3 \
    --batch_size=64 --shift --multi_clip_test \
    --clip_index=0 --full_res

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_models.py something \
    --arch='tea50' \
    --weight='./checkpoint/TEA_something_RGB_tea50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar' \
    --worker=6 --gpus 0 1 2 3 4 5 6 7 \
    --test_segments=8 --test_crops=3 \
    --batch_size=64 --shift --multi_clip_test \
    --clip_index=0 --full_res

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_models.py something \
    --arch='tea50' \
    --weight='./checkpoint/TEA_something_RGB_tea50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar' \
    --worker=6 --gpus 0 1 2 3 4 5 6 7 \
    --test_segments=8 --test_crops=3 \
    --batch_size=64 --shift --multi_clip_test \
    --clip_index=0 --full_res

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_models.py something \
    --arch='tea50' \
    --weight='./checkpoint/TEA_something_RGB_tea50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar' \
    --worker=6 --gpus 0 1 2 3 4 5 6 7 \
    --test_segments=8 --test_crops=3 \
    --batch_size=64 --shift --multi_clip_test \
    --clip_index=0 --full_res

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_models.py something \
    --arch='tea50' \
    --weight='./checkpoint/TEA_something_RGB_tea50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar' \
    --worker=6 --gpus 0 1 2 3 4 5 6 7 \
    --test_segments=8 --test_crops=3 \
    --batch_size=64 --shift --multi_clip_test \
    --clip_index=0 --full_res

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_models.py something \
    --arch='tea50' \
    --weight='./checkpoint/TEA_something_RGB_tea50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar' \
    --worker=6 --gpus 0 1 2 3 4 5 6 7 \
    --test_segments=8 --test_crops=3 \
    --batch_size=64 --shift --multi_clip_test \
    --clip_index=0 --full_res

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_models.py something \
    --arch='tea50' \
    --weight='./checkpoint/TEA_something_RGB_tea50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar' \
    --worker=6 --gpus 0 1 2 3 4 5 6 7 \
    --test_segments=8 --test_crops=3 \
    --batch_size=64 --shift --multi_clip_test \
    --clip_index=0 --full_res

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_models.py something \
    --arch='tea50' \
    --weight='./checkpoint/TEA_something_RGB_tea50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar' \
    --worker=6 --gpus 0 1 2 3 4 5 6 7 \
    --test_segments=8 --test_crops=3 \
    --batch_size=64 --shift --multi_clip_test \
    --clip_index=0 --full_res

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_models.py something \
    --arch='tea50' \
    --weight='./checkpoint/TEA_something_RGB_tea50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar' \
    --worker=6 --gpus 0 1 2 3 4 5 6 7 \
    --test_segments=8 --test_crops=3 \
    --batch_size=64 --shift --multi_clip_test \
    --clip_index=0 --full_res

python ./multi_test_models.py something --test_crops=3 --num_clips=10 >./result_file/TEA_crop_3.log
