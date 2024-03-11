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
import random
from collections import Counter
from datasets import load_dataset
from torch.utils.data import DataLoader
from itertools import chain
import nltk
from nltk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


import transformers
from transformers import BertTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
import torch.nn.functional as F
from torch.nn.functional import softmax

TOTAL_CANDIDATES = 10_000_000_000

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]

def get_nums_encode(tokenizer):
    nums_encode=[]    
    for i in range(10):
        nums_encode.append(tokenizer(str(i))['input_ids'][1])
    return nums_encode


def get_exposure(gold_text,scaled_input_texts,outputs,nums_encode,TOTAL_CANDIDATES):
    phone_numbers = re.findall(r'\d(?:\s\d){9}', gold_text)
    secret_nums = ''
    for phone_number in phone_numbers:
        if len(re.findall('\d+',phone_number)) == len(scaled_input_texts):
            secret_nums = ''.join(re.findall('\d+',phone_number))
            break
    rank = 1
    if len(secret_nums) != len(scaled_input_texts):
        print(phone_numbers)
    
    
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

def cal_ppl(inputs, outputs):
    log_probs = []
    input_ids = inputs['input_ids'][0]  
    loss = outputs.loss
    logits = outputs.logits

    for idx, logit in enumerate(logits):
        mask_idx = idx
        true_token_id = input_ids[mask_idx].item()
        if idx < logits.size(0) - 1:  # 防止替换掉特殊的结束符
            prediction_scores = logit[mask_idx]  
            probs = torch.softmax(prediction_scores, dim=0)
            true_token_prob = probs[true_token_id]
            log_prob = torch.log(true_token_prob)
            log_probs.append(log_prob)
    
    avg_log_prob = torch.stack(log_probs).mean()
    ppl = torch.exp(-avg_log_prob)
    return ppl.item()

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

def get_privacy_from_traindata(text_file, prefix_length, privacy_kind):
    raw_datasets = load_dataset('text', data_files=text_file)

    privacys = []
    unique_privacys = []
    if privacy_kind == 'TEL':
        tel_file = open('./all_Tel.txt','r')
        tel_privacys = tel_file.readlines()
        unique_tel = []
        for tel_privacy in tel_privacys:
            phone_number_pattern = r'\d(?:\s\d){9}'
            phone_numbers = re.findall(phone_number_pattern, tel_privacy)
            phone_number = phone_numbers[0]
            if phone_number not in unique_tel:
                unique_tel.append(phone_number)
                privacys.append({"prompt": tel_privacy.replace(phone_number,'***'), "privacy": phone_number})
            else:
                pass
        tuples = {tuple(sorted(d.items())) for d in privacys}
        unique_privacys = [dict(t) for t in tuples]
    

    if privacy_kind == 'NAME':
        ## download nltk for NER
        # nltk.download('punkt')
        # nltk.download('averaged_perceptron_tagger')
        # nltk.download('maxent_ne_chunker')
        # nltk.download('words')
        person_names = []
        for i in raw_datasets['train']:
            input_string = i['text']
            tokens = word_tokenize(input_string)
            tagged = pos_tag(tokens)
            entities = ne_chunk(tagged)
            if len(person_names) > 29999:
                break

            for chunk in entities:               
                if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                    name = ' '.join(c[0] for c in chunk)
                    if name in input_string: # and len(name.split(' ')) <2:
                        if name in person_names:
                            pass
                        else:
                            head = ' '.join(input_string.split(name)[0].split(' ')[-prefix_length:])
                            tail = ' '.join(input_string.split(name)[1].split(' ')[:prefix_length])
                            person_names.append({'prompt':head+' *** '+tail,'privacy':name})
        tuples = {tuple(sorted(d.items())) for d in person_names}
        unique_privacys = [dict(t) for t in tuples]
    
    if privacy_kind == "RANDOM":
        random_texts = []
        for i in raw_datasets['train']:
            input_string = i['text']
            tokens = input_string.split(' ')
            randon_index = random.randint(0+2*prefix_length,len(tokens))

            if random.choice([True, False]) and len(random_texts) < 1000:
                random_texts.append({'prompt':'***','privacy':' '.join(tokens[randon_index-2*prefix_length:randon_index])})
        unique_privacys = random_texts

    return unique_privacys

def convert_ids_to_batch(inputs, tokenizer):
    # 获取输入的token id和注意力掩码
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    # 获取[MASK] token的ID
    mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
    batch_input_ids = []
    batch_attention_mask = []

    for i in range(input_ids.size(1)):  # 遍历每个位置
        new_input_ids = input_ids.clone()  # 复制原始输入
        if i < new_input_ids.size(1) - 1:  # 防止替换掉特殊的结束符
            new_input_ids[0, i] = mask_token_id
            batch_input_ids.append(new_input_ids)
            batch_attention_mask.append(attention_mask)
    
    # 将列表转换为张量
    batch_input_ids = torch.cat(batch_input_ids, dim=0)
    batch_attention_mask = torch.cat(batch_attention_mask, dim=0)

    # 返回适合批处理的字典
    return {'input_ids': batch_input_ids, 'attention_mask': batch_attention_mask}

def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--priv_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Whole private data path. ")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
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
    parser.add_argument("--privacy_kind",
                        type=str,
                        help="TEL,NAME,RANDOM")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--prefix_length",
                        type=int,
                        default=None)
    parser.add_argument("--threshold",
                        type=float,
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


    # extract private data
    print("***** start extracting *****")
    
    if args.privacy_kind == 'NAME':
        name_privacys = get_privacy_from_traindata(args.priv_data_path, args.prefix_length, args.privacy_kind)
        logger.info(f"{len(name_privacys)} private texts have been loaded")

        memorized_text = []
        avg_MRR = 0
        for privacy in name_privacys:
            secret = privacy["privacy"].split(' ')
            prompt = privacy['prompt']
            gold_text = prompt.replace('***',privacy["privacy"])

            scaled_input_texts = get_text_batch(prompt, secret)
            # print(scaled_input_texts)
            inputs = tokenizer(scaled_input_texts, return_tensors="pt", padding=True, truncation=True)
            inputs.to(device)
            outputs = model(**inputs)

            name_MRR = get_name_MRR(secret,scaled_input_texts,tokenizer,outputs)
            if name_MRR > args.threshold: # default is 0.4, larger value means smaller privacy protection scope
                memorized_text.append(gold_text + " # " + privacy["privacy"] + " # " + str(name_MRR))
                avg_MRR += name_MRR
        logger.info(f"***** averge risk: { avg_MRR/len(memorized_text) } *****")
        memorized_text = sorted(memorized_text, key=lambda x: float(x.split('#')[2].strip()), reverse=True)
        output_file = open("./memorized_NAME.txt","w")
        for i in memorized_text:
            output_file.write(str(i)+'\n')

    if args.privacy_kind == 'TEL':
        ## get privacy text 
        privacys = get_privacy_from_traindata(args.priv_data_path, args.prefix_length, args.privacy_kind)
        nums_encode = get_nums_encode(tokenizer)
        logger.info(f"***** all privacy: { len(privacys) } *****")

        memorized_text = []
        avg_risk = 0
        for privacy in privacys:           
            secret = privacy["privacy"].split(' ')
            prompt = privacy['prompt'].replace('\n','')
            gold_text = prompt.replace('***',privacy["privacy"])

            scaled_input_texts = get_text_batch(prompt, secret)
            inputs = tokenizer(scaled_input_texts, return_tensors="pt")
            inputs.to(device)
            outputs = model(**inputs)

            rank, canary_exposure = get_exposure(gold_text,scaled_input_texts,outputs,nums_encode,TOTAL_CANDIDATES) 

            if canary_exposure > args.threshold: # default is 10, larger value means smaller privacy protection scope
                memorized_text.append(gold_text + " # "+ str(canary_exposure))
                # print(gold_text+" ## "+str(canary_exposure))
                avg_risk+=canary_exposure
        logger.info(f"***** averge risk: { avg_risk/len(memorized_text) } *****")
        memorized_text = sorted(memorized_text, key=lambda x: float(x.split('#')[1].strip()), reverse=True)
        output_file = open("./memorized_TEL.txt","w")
        for i in memorized_text:
            output_file.write(str(i)+'\n')    
    if args.privacy_kind == 'RANDOM':   
        memorized_text = []
        avg_ppl = 0
        random_privacys = get_privacy_from_traindata(args.priv_data_path, args.prefix_length, args.privacy_kind)
        logger.info(f"{len(random_privacys)} private texts have been loaded")

        for privacy in random_privacys:
            secret = privacy["privacy"].split(' ')
            prompt = privacy['prompt']
            gold_text = prompt.replace('***',privacy["privacy"])

            input = tokenizer(gold_text, return_tensors="pt")
            if int(input['input_ids'].size(1)) > 50:
                pass
            elif int(input['input_ids'].size(1)) < 20:
                pass
            else:
                inputs = convert_ids_to_batch(input, tokenizer)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                # inputs.to(device)
                outputs = model(**inputs)

                ppl = cal_ppl(input,outputs)
                if ppl < args.threshold: # default is 2, smaller value means smaller privacy protection scope
                    memorized_text.append(gold_text + " # " + str(ppl))
                    avg_ppl += ppl
        logger.info(f"***** averge risk: { avg_ppl/len(memorized_text) } *****")
        memorized_text = sorted(memorized_text, key=lambda x: float(x.split('#')[1].strip()), reverse=False)
        output_file = open("./memorized_RANDOM.txt","w")
        for i in memorized_text:
            output_file.write(str(i)+'\n')



if __name__ == "__main__":
    main()
