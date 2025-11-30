# HOPE-fix

train：
v1.2：
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
v1.2-cosine（75%+）：
python train_semi.py \

    --dataset ModelNet40 \

    --num_classes 40 \

    --n_labeled 800 \

    --batch_size 32 \

    --epochs 50 \

    --lr_img 5e-5 \

    --lr_pt 1e-4 \

    --weight_decay 1e-5 \

    --use_ma \

    --ma_warmup 3 \

    --ma_beta 0.85 \

    --use_low_confidence_correction \

    --weight_correction 1.0 \

    --use_sharpening \

    --sharpen_temp 0.5 \

    --sharpen_start 5 \

    --quantization_type swdc \

    --quantization_weight 0.1 \

    --weight_crc 1.0 \

    --weight_align 1.0 \

    --cps_weight 1.0 \

    --save ../checkpoints/ModelNet40_v1.2_Cosine_Final/ \

    --per_print 50



test：
python HOPE/test_semi.py \
    --dataset ModelNet40 \
    --n_labeled 800 \
    --quantization_type swdc \
    --gpu_id 1


    
<img width="797" height="801" alt="image" src="https://github.com/user-attachments/assets/88530846-d3d5-4e26-b760-6deb4e1c28e5" />
