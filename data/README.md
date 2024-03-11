1.Downloading Enron
Download raw Enron data from https://www.cs.cmu.edu/~enron/.
```
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
```

2.Processing Original Data

```
python preprocess_enron.py
```

The script will generate train, valid and test data under "./enron_data", of which there are 3000 valid and test data each.

Note: During the processing, in order to ensure that numeric type private data is tokenized to single digits, we process the private numeric data into a form segmented by " ".
```
"Freephone within the U.S.: 8773155218"   ==>   "Freephone within the U.S.: 8 7 7 3 1 5 5 2 1 8"
```

3.Finetuning BERT on Enron

Copy https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_no_trainer.py

Use the following command:
```
python -u accelerate_cli.py launch --multi_gpu \
    --num_processes 4 \
    --num_machines 1 \
    --main_process_port 12356 \
     run_mlm_no_trainer.py \
    --model_name_or_path bert-large-uncased \
    --train_file ./enron_data/train.txt \
    --validation_file ./enron_data/valid.txt \
    --config_name bert-large-uncased \
    --tokenizer_name bert-large-uncased \
    --num_train_epochs  10\
    --checkpointing_steps epoch \
    --per_device_train_batch_size 32 \
    --max_seq_length 256 \
    --output_dir ./model/ep10_bert_large
```

4.Getting Memorized Text

(1) Use following command to get memorized texts with private TEL:

```
python get_memorization.py \
    --model_name_or_path ./model/ep10_bert_large \
    --priv_data_path ./enron_data/train.txt \
    --gpus 6 \
    --max_seq_length 128 \
    --privacy_kind TEL \
    --prefix_length 10 \
    --threshold 10 \
```
The "threshold" of the phone number represents the risky exposure threshold. This value is used to control the scope of privacy protection. The larger the value, the smaller the scope of protection, which means that only those private texts with a high risk of leakage will be focused on. The memorized texts with TEL will be write in './memorized_TEL.txt'.

(2) Use following command to get memorized texts with private NAME:

```
python get_memorization.py \
    --model_name_or_path ./model/ep10_bert_large \
    --priv_data_path ./enron_data/train.txt \
    --gpus 6 \
    --max_seq_length 128 \
    --privacy_kind NAME \
    --prefix_length 10 \
    --threshold 0.4 \
```
The "threshold" of names represents the MRR threshold. The trend of this value is consistent with exposure of TEL. The memorized texts with NAME will be write in './memorized_NAME.txt'.

(3) Use following command to get memorized RANDOM texts:

```
python get_memorization.py \
    --model_name_or_path ./model/ep10_bert_large \
    --priv_data_path ./enron_data/train.txt \
    --gpus 6 \
    --max_seq_length 128 \
    --privacy_kind RANDOM \
    --prefix_length 10 \
    --threshold 2.5 \
```
The "threshold" of RANDOM represents the threshold of perplexity. The smaller the value, the smaller the scope of protection, which means that only those texts that the BERT model is more familiar with will be focused on. The memorized RANDOM texts will be write in './memorized_RANDOM.txt'.

5.Recording Original Privacy Leakage Risk

After running the above commands, we will obtain three memorized private TXT texts, and the privacy leakage risks in these texts will be recorded. 

In subsequent experiments, we will randomly select a portion of the private data from each TXT file for locating and editing privacy neurons. 

Then evaluating privacy leakage risks of edited BERT model in the respective private TXT texts.








