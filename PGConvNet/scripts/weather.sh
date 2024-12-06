python -u run.py --task_name imputation --is_training 1 --mask_rate 0.125 --model_id weather_mask_0.125 --model PGConvNet --root_path ./all_datasets/weather/ --data_path weather.csv --data custom --features M --seq_len 96 --label_len 0 --pred_len 96 --num_blocks 2 --large_size 3 --small_size 1 --kernel_size 3 --num_experts 2 --updim 32 --head_dropout 0.2 --enc_in 21 --dropout 0.1 --batch_size 16 --learning_rate 0.001 --train_epochs 100 --patience 10 --lradj type3 --des Exp --itr 1

python -u run.py --task_name imputation --is_training 1 --mask_rate 0.25 --model_id weather_mask_0.25 --model PGConvNet --root_path ./all_datasets/weather/ --data_path weather.csv --data custom --features M --seq_len 96 --label_len 0 --pred_len 96 --num_blocks 2 --large_size 3 --small_size 1 --kernel_size 3 --num_experts 2 --updim 32 --head_dropout 0.2 --enc_in 21 --dropout 0.1 --batch_size 16 --learning_rate 0.001 --train_epochs 100 --patience 10 --lradj type3 --des Exp --itr 1

python -u run.py --task_name imputation --is_training 1 --mask_rate 0.375 --model_id weather_mask_0.375 --model PGConvNet --root_path ./all_datasets/weather/ --data_path weather.csv --data custom --features M --seq_len 96 --label_len 0 --pred_len 96 --num_blocks 2 --large_size 3 --small_size 1 --kernel_size 3 --num_experts 2 --updim 32 --head_dropout 0.2 --enc_in 21 --dropout 0.1 --batch_size 16 --learning_rate 0.001 --train_epochs 100 --patience 10 --lradj type3 --des Exp --itr 1

python -u run.py --task_name imputation --is_training 1 --mask_rate 0.5 --model_id weather_mask_0.5 --model PGConvNet --root_path ./all_datasets/weather/ --data_path weather.csv --data custom --features M --seq_len 96 --label_len 0 --pred_len 96 --num_blocks 2 --large_size 3 --small_size 1 --kernel_size 3 --num_experts 2 --updim 32 --head_dropout 0.2 --enc_in 21 --dropout 0.1 --batch_size 16 --learning_rate 0.001 --train_epochs 100 --patience 10 --lradj type3 --des Exp --itr 1



