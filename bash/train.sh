python -u train.py \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --weight_decay 0.1 \
    --epochs 10 \
    --model_name "facebook/bart-large-cnn" \
    --model_path "model/model" \
    --checkpoint_path "model/checkpoint" \
    --data_path "data/speeches/speeches_selected"
