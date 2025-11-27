# HOPE-fix

运行命令：
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
