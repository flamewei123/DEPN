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
Please follow the README in DEPN/data to obtain "priv_data_path" and "model_name_or_path". And batch_size is the number of approximation steps from Eq.4 in our paper.

**2.Aggregate** privacy neurons by filtering neurons

running the following command:
```

```
