CUDA_VISIBLE_DEVICES=0 python train.py \
    --input_wavs_dir ./  \
    --input_training_file ./filelists/ljs_audio_text_train_filelist.txt   \
    --input_validation_file ./filelists/ljs_audio_text_val_filelist.txt \
    --config configs/config.json   \
    --checkpoint_path ./logs_training
