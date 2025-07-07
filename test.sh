results_path="./test"  # replace to your results path
batch_size=4
weight_path="/data/private/ly/CLIP/0704_all/savedir/checkpoint50.pt"

TASK="BCMA"

CUDA_VISIBLE_DEVICES="0" python ./unimol/test.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 2 --ddp-backend=c10d --batch-size $batch_size \
       --task peptideclip --loss peptideclip --arch peptideclip  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 \
       --test-task $TASK \