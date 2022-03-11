### evaluation, run through a dataset
python3 main.py --save_dir ./eval/CUFED/TTSR_multitest \
               --reset True \
               --log_file_name eval.log \
               --eval_save_results False \
               --num_workers 1 \
               --dataset CUFED \
               --dataset_dir ./CUFED/ \
               --model_path ./train/CUFED/TTSR-rec-3090/model/model_00200.pt \
               --eval_multiframe_count 5 \
                --eval_multiframe True \
#               --randompick True