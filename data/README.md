**1.Downloading Enron**
Download raw Enron data from https://www.cs.cmu.edu/~enron/.
```
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
```

**2.Processing Original Data**

```
python preprocess_enron.py
```

The script will generate train, valid and test data under "./enron_data", of which there are 3000 valid and test data each.

**Note:** During the processing, in order to ensure that numeric type private data is tokenized to single digits, we process the private numeric data into a form segmented by " ".
```
"Freephone within the U.S.: 8773155218"   ==>   "Freephone within the U.S.: 8 7 7 3 1 5 5 2 1 8"
```

**3.Finetuning BERT on Enron**

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

**4.Getting Memorized Text**

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
```
03/01/2024 15:12:32 - INFO - __main__ -   ***** averge risk: 16.4272194064181782 *****
```

(2) Use following command to get memorized texts with private NAME:

```
python get_memorization.py \
    --model_name_or_path ./model/ep10_bert_large \
    --priv_data_path ./enron_data/train.txt \
    --gpus 6 \
    --max_seq_length 128 \
    --privacy_kind NAME \
    --prefix_length 20 \
    --threshold 0.4 \
```
The "threshold" of names represents the MRR threshold. The trend of this value is consistent with exposure of TEL. The memorized texts with NAME will be write in './memorized_NAME.txt'.
```
03/01/2024 15:45:20 - INFO - __main__ -   ***** averge risk: 0.8093582172346347 *****
```

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
```
03/01/2024 15:50:01 - INFO - __main__ -   ***** averge risk: 2.3234634709358217 *****
```

**5.Recording Original Privacy Leakage Risk**

**Note:** The above values of privacy leakage risk are only references. 

The specific risk values will change according to factors such as "model_size", "num_train_epochs", "prefix_length", "threshold", etc. 

Please fine-tune a privacy BERT model according to 3 and then evaluate privacy leakage risks by yourself.

The following are the privacy leak risks of BERT-F on different privacy types of data

| Privacy Type | Metric | Value |
|:--------:|:--------:|:--------:|
| Phone Number | Exposure | 16.43 |
| Name | MRR | 0.81 |
| Random Text | PPL | 2.32 |

In subsequent experiments, we will randomly select a portion of the private data from each TXT file for locating and editing privacy neurons. 

```
import random

def random_lines_to_file(data_path, target_path, num):

    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        num = min(num, len(lines))
        
        selected_lines = random.sample(lines, num)
        
        with open(target_path, 'w', encoding='utf-8') as file:
            file.writelines(selected_lines)
        
        print(f"save {num} lines in '{target_path}'。")
    except FileNotFoundError:
        print(f"file'{data_path}'data_path does not exist")
    except Exception as e:
        print(f"Error: {e}")


data_path = './memorized_RANDOM.txt'   # './memorized_TEL.txt' './memorized_NAME.txt'
target_path = './sampled_RANDOM.txt'  
num = 100  

random_lines_to_file(data_path, target_path, num)
```

Then evaluating the changes of privacy leakage risks from edited BERT model in the private texts.








