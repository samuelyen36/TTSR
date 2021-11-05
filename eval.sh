### evaluation
python main.py --save_dir ./eval/CUFED/TTSR_record_lv5 \
               --reset True \
               --log_file_name eval.log \
               --eval True \
               --eval_save_results True \
               --num_workers 4 \
               --dataset CUFED \
               --dataset_dir ./CUFED/ \
               --model_path ./train/CUFED/TTSR-rec-3090/model/model_00200.pt \
               --eval_ref 5 \
#               --randompick True