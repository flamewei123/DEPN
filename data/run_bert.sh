python -u accelerate_cli.py launch --multi_gpu \
    --num_processes 4 \
    --num_machines 1 \
    --main_process_port 12356 \
     run_mlm_no_trainer.py \
    --model_name_or_path /data/wuxinwei/transformers/model/2023/0329_test_mlm/epoch_0/ \
    --train_file /home/wuxinwei/wuxinwei_projects/unfair_privacy/DEPN/data/enron_data/train.txt \
    --validation_file /home/wuxinwei/wuxinwei_projects/unfair_privacy/DEPN/data/enron_data/valid.txt \
    --num_train_epochs  5\
    --checkpointing_steps epoch \
    --per_device_train_batch_size 256 \
    --max_seq_length 128 \
    --output_dir /data/wuxinwei/transformers/model/2024/0305_ep10_bert_base
