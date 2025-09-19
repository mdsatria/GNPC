python pipeline/generate_morphology.py \
--csv_slide="sample_data/csv/dataset.csv" \
--dir_wsi="/mnt/storage/datasets/npc/cuhk/slide_he" \
--dir_nuclei="/mnt/storage/project_data/nuclei_prediction/hovernet_pkl/cuhk" \
--dir_patch="sample_data/patches/cuhk/coords" \
--dir_save="sample_data/morphology/cuhk" \
--log_file="sample_data/morphology/logs/cuhk.log" \
--target_mpp=0.5


