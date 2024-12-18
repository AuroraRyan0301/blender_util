MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
TRAIN_FLAGS="--lr 2e-5 --batch_size 1"

python scripts/image_train.py --data_dir /data/ris_test $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS