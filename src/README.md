Protecting private data through DEPN is divided into three steps: detect privacy neurons, aggregate privacy neurons, and edit privacy neurons.

**1.Detect** privacy neurons by privacy attribution scores

running the following command:
```
python 1_calculate_attribution.py \
    --model_name_or_path ../data/model/ep10_bert_large \
    --priv_data_path ../data/sampled_TEL.json \
    --output_dir ../Enorn_tel_bert_large_ep10_results/ \
    --output_prefix Enorn_tel_bert_large_ep10 \
    --gpus 4 \
    --max_seq_length 128 \
    --batch_size 20 \
    --num_batch 1 \
```
Please follow the README in DEPN/data to obtain "priv_data_path" and "model_name_or_path". 

And "batch_size" is the number of approximation steps from Eq.4 in our paper, the larger "batch_size, the more accurate the attribution score will be, but the time cost will increase.

**2.Aggregate** privacy neurons by filtering neurons

running the following command:
```
python 2_get_kn.py \ 
../Enorn_tel_bert_base_ep10_results/ \
0.1 \
0.5
```
0.1 is the "threshold_ratio", we filter the neurons whose attribution score is less than threshold_ratio * the maximum value. 

0.5 is the "mode_ratio_bag", we filter out neurons whose frequency is less than mode_ratio_bag in the same batch of text.

**3.Edit** privacy neurons by earsing neurons

running the following command:
```
python 3_edit_privacy_neurons.py \
    --model_name_or_path ../data/model/2024/0305_ep10_bert_base \
    --priv_data_path ../data/memorized_TEL.txt \
    --validation_path ../data/enron_data/valid.txt \
    --kn_dir ../Enorn_tel_bert_base_ep10_results/kn/kn_bag-Enorn_tel_bert_base_ep10.json \
    --gpus 6 \
    --max_seq_length 128 \
    --erase_kn_num 200 \
    --do_random_kn False \
    --input_prefix TEL \
```
"erase_kn_num" is the number of edited neurons, the larger "erase_kn_num", the better the privacy protection performance, but the greater the model performance will be damaged.

When "do_random_kn" is True, the edited neurons will be randomly assigned for xiaorongshiyan
