results_path="/data/private/ly/CLIP/test_set/bcma/result"  # replace to your results path
batch_size=8
weight_path="/data/private/ly/CLIP/0613/savedir/checkpoint_best.pt" # 只需要checkpoint就可以了
Q_POCKET_PATH="/data/private/ly/CLIP/test_set/bcma/receptor.lmdb" # path to the molecule file
T_POCKET_PATH="/data/private/ly/CLIP/test_set/bcma/petide.lmdb" # path to the pocket file
EMB_DIR="/data/private/ly/CLIP/test_set/bcma/emb" # path to the cached mol embedding file

CUDA_VISIBLE_DEVICES="1" python ./unimol/retrieval.py --user-dir ./unimol "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task peptideclip --loss peptideclip --arch peptideclip  \
       --max-pocket-atoms 256 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --query-pocket-path $Q_POCKET_PATH \
       --target-pocket-path $T_POCKET_PATH \
       --emb-dir $EMB_DIR \