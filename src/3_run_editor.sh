python 3_edit_privacy_neurons.py \
    --model_name_or_path ../data/model/2024/0305_ep10_bert_base \
    --priv_data_path ../data/memorized_TEL.txt \
    --validation_path ../data/enron_data/valid.txt \
    --kn_dir ../Enorn_tel_bert_base_ep10_results/kn/kn_bag-Enorn_tel_bert_base_ep10.json \
    --gpus 6 \
    --max_seq_length 128 \
    --erase_kn_num 20 \
    --do_random_kn False \
    --input_prefix TEL \
