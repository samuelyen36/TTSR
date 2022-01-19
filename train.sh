### training TTSR
#python3 main.py --save_dir ./train/CUFED/TTSR_lrSR \
#               --reset True \
#               --log_file_name train.log \
#               --num_gpu 1 \
#               --num_workers 9 \
#               --dataset CUFED \
#               --dataset_dir ./CUFED/ \
#               --n_feats 64 \
#               --lr_rate 1e-4 \
#               --lr_rate_dis 1e-4 \
#               --lr_rate_lte 1e-5 \
#               --rec_w 1 \
#               --per_w 1e-2 \
#               --tpl_w 1e-2 \
#               --adv_w 1e-3 \
#               --batch_size 9 \
#               --num_init_epochs 10 \
#               --num_epochs 150 \
#               --print_every 600 \
#               --save_every 30 \
#               --val_every 3    
#               --load_at_training True \
#               --resume_weight train/CUFED/TTSR_withGANloss/model/model_00200.pt


 ### training TTSR-rec

 python3 main.py --save_dir ./train/CUFED/testtesttest \
                --reset True \
                --log_file_name train.log \
                --num_gpu 1 \
                --num_workers 9 \
                --dataset CUFED \
                --dataset_dir ./CUFED/ \
                --n_feats 64 \
                --lr_rate 1e-4 \
                --lr_rate_dis 1e-4 \
                --lr_rate_lte 1e-5 \
                --rec_w 1 \
                --per_w 0 \
                --tpl_w 0 \
               --adv_w 0 \
                --batch_size 18 \
                --num_init_epochs 0 \
                --num_epochs 300 \
                --print_every 600 \
                --save_every 50 \
                --val_every 1   \
                --featurelevel 1
