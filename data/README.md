1.Download Enron
Download raw Enron data from https://www.cs.cmu.edu/~enron/.
```
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
```

2.Process Data

```
python preprocess_enron.py
```

The script will generate train, valid and test data under "./enron_data", of which there are 3000 valid and test data each.

Note: During the processing, in order to ensure that numeric type private data is tokenized to single digits, we process the private numeric data into a form segmented by " ".
```
"Freephone within the U.S.: 8773155218"   ==>   "Freephone within the U.S.: 8 7 7 3 1 5 5 2 1 8"
```

3.Finetune BERT on Enron

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

4.Get memorized Text
