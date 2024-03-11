"""
BERT MLM runner
"""

import logging
import argparse
import math
import os
import torch
import re
import random
import numpy as np
import json, jsonlines
import pickle
import time
import random
from collections import Counter
from datasets import load_dataset
from torch.utils.data import DataLoader
from itertools import chain

import transformers
from transformers import BertTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from custom_bert import BertForMaskedLM
import torch.nn.functional as F

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_evaldata(eval_data_path, tokenizer):

    data_files = {}
    data_files["validation"] = eval_data_path
    raw_datasets = load_dataset('text', data_files=data_files)

    text_column_name = 'text'
    max_seq_length = 128

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column_name],
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=False,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )

    eval_dataset = tokenized_datasets["validation"]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=8)

    return eval_dataloader

def eval_ppl(eval_dataloader,device,model):
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch.to(device)
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss.repeat(8))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    return perplexity


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]

def get_nums_encode(tokenizer):
    nums_encode=[]    
    for i in range(10):
        nums_encode.append(tokenizer(str(i))['input_ids'][1])
    return nums_encode

def load_privacys(unique_priv_path):
    with open(unique_priv_path,'r') as file:
        lines = file.readlines()
        unique_priv_list = []
        cnt = 1
        for i in lines:
            keys = i.strip().split('# ')
            unique_priv_list.append(keys)
            cnt += 1

    return unique_priv_list


def get_exposure(gold_text,scaled_input_texts,outputs,nums_encode,TOTAL_CANDIDATES):
    secret_nums = ''.join(re.findall('\d+',gold_text))
    rank = 1
    
    for i in range(len(secret_nums)):
        input_tokens = scaled_input_texts[i].split(' ')
        target_pos = input_tokens.index('[MASK]')
        rank += get_tar_rank(secret_nums[i],outputs.logits[i][target_pos+1],nums_encode)*pow(10,len(secret_nums)-1-i)
    canary_exposure = math.log(TOTAL_CANDIDATES, 2) - math.log(rank, 2)

    return rank, canary_exposure

def get_tar_rank(target,output,nums_encode):
    values = {}
    for i in range(len(nums_encode)):
        values[i] = float(output[nums_encode[i]])
    ranked_vals = dict(sorted(values.items(), key=lambda x: x[1],reverse = True))           
    rank = list(ranked_vals.keys()).index(int(target))

    return rank

def get_text_batch(prompt, secret):
    scaled_input_texts = []
    for i in range(len(secret)):
        masked_secret = ['[MASK]' if i == j else secret[j] for j in range(len(secret))]
        # masked_secret = '[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]'.split(' ')
        scaled_input_texts.append(prompt.replace('***',' '.join(masked_secret)))
    return scaled_input_texts

def get_name_MRR(secret,scaled_input_texts,tokenizer,outputs):
    res = 0
    for i in range(len(secret)):
        encode_token = tokenizer(str(i))['input_ids'][1]
        input_tokens = scaled_input_texts[i].split(' ')
        target_pos = input_tokens.index('[MASK]')
        # print(len(outputs.logits[i]))
        if target_pos >= len(outputs.logits[i]):
            return 0
        # pred_prob = F.softmax(outputs.logits, dim=1)
        else:
            sorted_indices = torch.argsort(outputs.logits[i][target_pos], descending=True)
            rank = (sorted_indices == encode_token).nonzero().item() + 1
        res+=1/rank        

    return res/len(secret)

def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--priv_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Whole private data path. ")
    parser.add_argument("--validation_path",
                        default=None,
                        type=str,
                        help="validation data to evaluate the fine-tuned model ")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_random_kn",
                        default=False,
                        type=bool,
                        help="use random position kn")
    parser.add_argument("--kn_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where important positions are stored.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--erase_kn_num",
                        default=10,
                        type=int,
                        help="how many kn to erase.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="available gpus id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--input_prefix",
                        type=str,
                        default=None)

    # parse arguments
    args = parser.parse_args()

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device("cuda:%s" % args.gpus)
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load pre-trained BERT
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    with open(args.kn_dir, 'r') as fr:
        kn_bag_list = json.load(fr)


    # # ======================== eval ori model =================================  
    eval_data_path = args.validation_path
    eval_dataloader = load_evaldata(eval_data_path, tokenizer)

    print('start evaluating original model')
    print(f"perplexity: {eval_ppl(eval_dataloader,device,model)}")

    # =========== get ori exposure  ==============

    txt_type = args.input_prefix
    if txt_type == 'TEL':

        TOTAL_CANDIDATES = 10_000_000_000

        unique_privacys = load_privacys(args.priv_data_path)

        nums_encode = get_nums_encode(tokenizer)
        before_exp_results = []
        after_exp_results = []
        prompt = ''
        exp_sum = 0
        count = 0

        for privacy in unique_privacys:      
            gold_text = privacy[0] 
            secret = re.findall(r'\d(?:\s\d){9}', gold_text)[0]
            prompt = gold_text.replace(secret,'***')
            
            scaled_input_texts = get_text_batch(prompt, secret.split(' ')) 
            inputs = tokenizer(scaled_input_texts, return_tensors="pt")
            inputs.to(device)
            outputs = model(**inputs) # (secret_length,input_length,vocab_size) # (10,16,30522)

            rank, canary_exposure = get_exposure(gold_text,scaled_input_texts,outputs,nums_encode,TOTAL_CANDIDATES)   

            single_priv = {   
                'sceret':gold_text,
                'rank': rank,
                'exp': canary_exposure
                }
        
            before_exp_results.append(single_priv)
            # print('$secret: ',' '.join(secret),'$, training_count: '+str(privacy[1])+', rank: '+str(rank)+', exp: '+str(canary_exposure))
            exp_sum += canary_exposure
            count += 1
        print('#'*30)
        print(prompt, ' average exp: ',exp_sum/count)

            
        # ======================== erase privacy neurons =================================
        with open(args.kn_dir, 'r') as fr:
            kn_bag_list = json.load(fr)

        kn_rel = []
        kn_counter = Counter()
        for kn_bag in kn_bag_list:
            for kn in kn_bag:
                kn_counter.update([pos_list2str(kn)])
        most_common_kn = kn_counter.most_common(args.erase_kn_num)
        
        ### change random kn
        if args.do_random_kn:
            random_kn = []
            for i in most_common_kn:
                # layer = random.randint(0,config.num_hidden_layers-1)
                # pos = random.randint(0, config.intermediate_size-1)
                layer = i[0].split('@')[0]
                # pos = int(i[0].split('@')[1])+random.randint(1,10)
                pos = random.randint(0, config.intermediate_size-1)
                random_kn.append(tuple([str(layer) + '@' + str(pos), i[1]]))
            most_common_kn = random_kn

        print('## erased kn:',most_common_kn)
        print('## erased kn num:',len(most_common_kn))
        kn_rel = [pos_str2list(kn_str[0]) for kn_str in most_common_kn]


        for layer, pos in kn_rel:
            with torch.no_grad():
                unk_emb = model.bert.embeddings.word_embeddings.weight[100]
                # model.bert.encoder.layer[layer].output.dense.weight[:, pos] = unk_emb
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] = 0
        print('start evaluating erased model')


        # ======================== eval new model  =================================
        print(f"perplexity: {eval_ppl(eval_dataloader,device,model)}")



        # =========== get new exposure  ==============
        count = 0
        exp_sum = 0
        for privacy in unique_privacys:        
            gold_text = privacy[0] 
            secret = re.findall(r'\d(?:\s\d){9}', gold_text)[0]
            prompt = gold_text.replace(secret,'***')
            
            scaled_input_texts = get_text_batch(prompt, secret.split(' ')) 
            inputs = tokenizer(scaled_input_texts, return_tensors="pt")
            inputs.to(device)
            outputs = model(**inputs) # (secret_length,input_length,vocab_size) # (10,16,30522)

            rank, canary_exposure = get_exposure(gold_text,scaled_input_texts,outputs,nums_encode,TOTAL_CANDIDATES)   

            single_priv = {   
                'sceret':gold_text,
                'rank': rank,
                'exp': canary_exposure
                }
        
            after_exp_results.append(single_priv)
            # print('$secret: ',' '.join(secret),'$, training_count: '+str(privacy[1])+', rank: '+str(rank)+', exp: '+str(canary_exposure))
            exp_sum += canary_exposure
            count += 1
        print('#'*30)
        print(prompt, ' average exp: ',exp_sum/count)

        #### show significant results
        # sum1, sum2, count = 0,0,0
        # for i in range(len(after_exp_results)):
        #     if  before_exp_results[i]['exp'] - after_exp_results[i]['exp'] >2:
        #         print('$secret: ',before_exp_results[i]['sceret'], \
        #             ', before_exp: ',before_exp_results[i]['exp'],  \
        #             ', after_exp: ',after_exp_results[i]['exp'])
        #         sum1 += before_exp_results[i]['exp']
        #         sum2 += after_exp_results[i]['exp']
        #         count += 1
        # print(sum1/count)
        # print(sum2/count)



    if txt_type == 'NAME':
        # =========== get ori MRR  ==============

        unique_privacys = load_privacys(args.priv_data_path)
        before_exp_results = []
        after_exp_results = []
        MRR_sum = 0
        count = 0  

        for privacy in unique_privacys:        
            secret = privacy[1].split(' ')
            gold_text = privacy[0] 
            prompt = gold_text.replace(' '.join(secret),'***')
            
            scaled_input_texts = get_text_batch(prompt, secret) # list of input texts (secret_length,input_length)
            inputs = tokenizer(scaled_input_texts, return_tensors="pt", padding=True, truncation=True)
            inputs.to(device)
            outputs = model(**inputs) # (secret_length,input_length,vocab_size)

            name_MRR = get_name_MRR(secret,scaled_input_texts,tokenizer,outputs)

            single_priv = {   
                'sceret':privacy[1],
                'text':gold_text,
                'MRR': name_MRR,
                }
            
            MRR_sum += name_MRR
            count += 1
            before_exp_results.append(single_priv)
            # print('$secret: ',' '.join(secret),'$, training_count: '+str(privacy[1])+', rank: '+str(rank)+', exp: '+str(canary_exposure))
        print('#'*30)
        print('average pred: ',MRR_sum/count)

            
        # ======================== erase privacy neurons =================================
        with open(args.kn_dir, 'r') as fr:
            kn_bag_list = json.load(fr)

        kn_rel = []
        kn_counter = Counter()
        for kn_bag in kn_bag_list:
            for kn in kn_bag:
                kn_counter.update([pos_list2str(kn)])
        most_common_kn = kn_counter.most_common(args.erase_kn_num)
        
        ### change random kn
        if args.do_random_kn:
            random_kn = []
            for i in most_common_kn:
                layer = random.randint(0,config.num_hidden_layers-1)
                pos = random.randint(0, config.intermediate_size-1)
                random_kn.append(tuple([str(layer) + '@' + str(pos), i[1]]))
            most_common_kn = random_kn

        print('## erased kn:',most_common_kn)
        print('## erased kn num:',len(most_common_kn))
        kn_rel = [pos_str2list(kn_str[0]) for kn_str in most_common_kn]


        for layer, pos in kn_rel:
            with torch.no_grad():
                unk_emb = model.bert.embeddings.word_embeddings.weight[100]
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] = unk_emb
                #model.bert.encoder.layer[layer].output.dense.weight[:, pos] = 0
        print('start evaluating erased model')


        # ======================== eval new model  =================================
        print(f"perplexity: {eval_ppl(eval_dataloader,device,model)}")

        # =========== get new MRR  ==============
        MRR_sum = 0
        count = 0  

        for privacy in unique_privacys:        
            secret = privacy[1].split(' ')
            gold_text = privacy[0] 
            prompt = gold_text.replace(' '.join(secret),'***')
            
            scaled_input_texts = get_text_batch(prompt, secret) # list of input texts (secret_length,input_length)
            inputs = tokenizer(scaled_input_texts, return_tensors="pt", padding=True, truncation=True)
            inputs.to(device)
            outputs = model(**inputs) # (secret_length,input_length,vocab_size)

            name_MRR = get_name_MRR(secret,scaled_input_texts,tokenizer,outputs)

            single_priv = {   
                'sceret':privacy[1],
                'text':gold_text,
                'MRR': name_MRR,
                }
            
            MRR_sum += name_MRR
            count += 1
            after_exp_results.append(single_priv)
            # print('$secret: ',' '.join(secret),'$, training_count: '+str(privacy[1])+', rank: '+str(rank)+', exp: '+str(canary_exposure))
        print('#'*30)
        print('average pred: ',MRR_sum/count)

        ### show positive and negative results
        # for i in range(len(after_exp_results)):
        #     if  before_exp_results[i]['pred'] < after_exp_results[i]['pred']:
        #         print('$secret: ',before_exp_results[i]['sceret'], \
        #             #'$, text: ',after_exp_results[i]['text'],  \
        #             ', before_pred: ',before_exp_results[i]['pred'],  \
        #             ', after_pred: ',after_exp_results[i]['pred'])
        # print('#'*30)
        # for i in range(len(after_exp_results)):
        #     if after_exp_results[i]['pred'] < before_exp_results[i]['pred']:
        #         print('$secret: ',before_exp_results[i]['sceret'], \
        #             #'$, text: ',after_exp_results[i]['text'],  \
        #             ', before_pred: ',before_exp_results[i]['pred'],  \
        #             ', after_pred: ',after_exp_results[i]['pred'])
                

    if txt_type == 'RANDOM':
        # =========== get ori ppl  ==============

        priv_dataloader = load_evaldata(args.priv_data_path, tokenizer)
        print(f"txt_ppl: : {eval_ppl(priv_dataloader,device,model)}")

        # ======================== erase privacy neurons =================================
        with open(args.kn_dir, 'r') as fr:
            kn_bag_list = json.load(fr)

        kn_rel = []
        kn_counter = Counter()
        for kn_bag in kn_bag_list:
            for kn in kn_bag:
                kn_counter.update([pos_list2str(kn)])
        most_common_kn = kn_counter.most_common(args.erase_kn_num)
        
        ### change random kn
        if args.do_random_kn:
            random_kn = []
            for i in most_common_kn:
                layer = random.randint(0,config.num_hidden_layers-1)
                pos = random.randint(0, config.intermediate_size-1)
                random_kn.append(tuple([str(layer) + '@' + str(pos), i[1]]))
            most_common_kn = random_kn

        print('## erased kn:',most_common_kn)
        print('## erased kn num:',len(most_common_kn))
        kn_rel = [pos_str2list(kn_str[0]) for kn_str in most_common_kn]


        for layer, pos in kn_rel:
            with torch.no_grad():
                unk_emb = model.bert.embeddings.word_embeddings.weight[100]
                model.bert.encoder.layer[layer].output.dense.weight[:, pos] = unk_emb
                #model.bert.encoder.layer[layer].output.dense.weight[:, pos] = 0
        print('start evaluating erased model')

        # =========== get new ppl  ==============
        print(f"txt_ppl: : {eval_ppl(priv_dataloader,device,model)}")


if __name__ == "__main__":
    main()
