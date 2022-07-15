#conda activate jittor

CUDA_VISIBLE_DEVICES=0 python test.py --data_path /home/pug/jittor-competition/dataset \
--output_path ./results/
