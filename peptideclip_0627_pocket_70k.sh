data_path="/data/private/ly/new_clip_bidirect/70k/lmdb"
save_dir="/data/private/ly/CLIP/0627_70k/savedir"
tmp_save_dir="/data/private/ly/CLIP/0627_70k/tmp_save_dir"
tsb_dir="/data/private/ly/CLIP/0627_70k/tsb_dir"

n_gpu=2
MASTER_PORT=10061
finetune_pocket1_model="/data/private/ly/CLIP/unimol_model/pocket_pre_220816.pt" # unimol pretrained pocket model
finetune_pocket2_model="/data/private/ly/CLIP/unimol_model/pocket_pre_220816.pt" # unimol pretrained pocket model


batch_size=16
batch_size_valid=16
epoch=1000
dropout=0.4
weight_decay=0.3
update_freq=4
dist_threshold=8.0
recycling=3
lr=5e-4

warmup=0.1

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="0,3" torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task peptideclip --loss peptideclip --arch peptideclip  \
       --max-pocket-atoms 256 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 --weight-decay $weight_decay \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 1 \
       --tensorboard-logdir $tsb_dir \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric valid_pocket_bedroc --patience 500 --all-gather-list-size 2048000 \
       --save-dir $save_dir --tmp-save-dir $tmp_save_dir --save-interval 50 \
       --find-unused-parameters \
       --maximize-best-checkpoint-metric \
       --finetune-pocket1-model $finetune_pocket1_model \
       --finetune-pocket2-model $finetune_pocket2_model \
       --skip-invalid-size-inputs-valid-test
#注意best-checkpoint-metric改过名字