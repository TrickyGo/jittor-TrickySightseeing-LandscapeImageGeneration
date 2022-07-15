#conda activate jittor

CUDA_VISIBLE_DEVICES=0 python train.py --data_path /home/pug/jittor-competition/dataset \
--output_path ./results/ \
--batch_size 10 \
--epoch 0 \
--n_epochs 611
