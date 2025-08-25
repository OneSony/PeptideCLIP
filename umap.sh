model_name="0819_combine_normal"

CUDA_VISIBLE_DEVICES="7" python /data/private/ly/PeptideCLIP/unimol/inference.py \
    --model_path /data/private/ly/CLIP/$model_name/savedir/checkpoint300.pt \
    --data_path /data/private/ly/PeptideCLIP/dataset/test/BCMA/BCMA.lmdb \
    --dict_path /data/private/ly/new_clip_bidirect/combine/dict_pkt.txt \
    --save_results /data/private/ly/CLIP_results/$model_name/300/BCMA/
    #--create_umap \
    #--umap_save_path /data/private/ly/CLIP_results/trop2_all.png