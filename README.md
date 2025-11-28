# HOPE-fix

train：
python HOPE/train_semi.py \
    --dataset ModelNet40 \
    --num_classes 40 \
    --n_labeled 800 \
    --use_ma \
    --use_low_confidence_correction \
    --ma_warmup 3 \
    --ma_beta 0.8 \
    --weight_correction 1 \
    --batch_size 32 \
    --epochs 50 \
    --save ./checkpoints/ModelNet40_v1.1_weighted/

test：
python HOPE/test_semi.py \
    --dataset ModelNet40 \
    --n_labeled 800 \
    --quantization_type swdc \
    --gpu_id 1
<img width="797" height="801" alt="image" src="https://github.com/user-attachments/assets/88530846-d3d5-4e26-b760-6deb4e1c28e5" />
