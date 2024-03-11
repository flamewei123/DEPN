"""
BERT MLM runner
"""

import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time

import transformers
from transformers import AutoModelForMaskedLM, AutoConfig, BertTokenizer, AutoTokenizer
from custom_bert import BertForMaskedLM
import torch.nn.functional as F

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def example2feature(example, max_seq_length, tokenizer):
    """
    Convert an example into input features
    eaxmple: ['Frederik Kaiser works in the field of [MASK].', 'astronomy', 'P101(field of work)']
    """
    features = []
    tokenslist = []

    ori_tokens = tokenizer.tokenize(example[0])
    # All templates are simple, almost no one will exceed the length limit.
    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[:max_seq_length - 2]

    # add special tokens
    tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    # Generate id and attention mask
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Pad [PAD] tokens (id in BERT-base-cased: 0) up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    segment_ids += padding
    input_mask += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    }
    tokens_info = {
        "tokens":tokens,
        "gold_obj":example[1],
        "pred_obj": None
    }
    return features, tokens_info


def scaled_input(emb, batch_size, num_batch):
    # emb: (1, ffn_size)
    baseline = torch.zeros_like(emb)  # (1, ffn_size)

    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    ig_triplet = []
    ig = np.array(ig_list)  # 12, 3072
    max_ig = ig.max()
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
            #ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet

def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--priv_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be .json file for the MLM task. ")
    parser.add_argument("--model_name_or_path",
                        type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models.",
                        required=False,
    )
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_prefix",
                        default=None,
                        type=str,
                        required=True,
                        help="The output prefix to indentify each running of experiment. ")

    # Other parameters
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
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument( "--use_slow_tokenizer",
                        action="store_true",
                        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    # parameters about integrated grad
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for cut.")
    parser.add_argument("--num_batch",
                        default=10,
                        type=int,
                        help="Num batch of an example.")

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
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # save args
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, args.output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

    # init tokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=args.do_lower_case)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    # Load pre-trained BERT
    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    model.to(device)

    # data parallel
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # prepare eval set
    with open(args.priv_data_path, 'r') as f:
        eval_bag_list_all = json.load(f)
    eval_bag_list_perrel = []
    for bag_idx, eval_bag in enumerate(eval_bag_list_all):
        eval_bag_list_perrel.append(eval_bag)


    # evaluate each privacy text
    # record running time
    tic = time.perf_counter()
    print('start processing, while dataset is ',len(eval_bag_list_perrel))
    count = 0
    with jsonlines.open(os.path.join(args.output_dir, args.output_prefix + '.priv' + '.jsonl'), 'w') as fw:
        for bag_idx, priv_texts in enumerate(eval_bag_list_perrel):
            res_dict_bag = []

            sum_list = [[0 for i in range(config.intermediate_size)] for j in range(config.num_hidden_layers)]
            _, tokens_info = example2feature(priv_texts[0], args.max_seq_length, tokenizer)
        
            for eval_example in priv_texts:
                #print(eval_example)
                eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)
                # convert features to long type tensors
                input_ids, input_mask, segment_ids = eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)

                # record real input length
                input_len = int(input_mask[0].sum())

                # record [MASK]'s position
                tgt_pos = tokens_info['tokens'].index('[MASK]')

                # record various results
                res_dict = {
                    'ig_gold': [],
                }

                for tgt_layer in range(model.bert.config.num_hidden_layers):
                    ffn_weights, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)  # (1, ffn_size), (1, n_vocab)
                    pred_label = int(torch.argmax(logits[0, :]))  # scalar
                    gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                    # tokens_info['pred_obj'] = tokenizer.convert_ids_to_tokens(pred_label)
                    # print(pred_label,'#',gold_label)
                    scaled_weights, weights_step = scaled_input(ffn_weights, args.batch_size, args.num_batch)  # (num_points, ffn_size), (ffn_size)
                    scaled_weights.requires_grad_(True)

                    # integrated grad at the gold label for each layer
                    ig_gold = None
                    for batch_idx in range(args.num_batch):
                        batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                        _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=gold_label)  # (batch, n_vocab), (batch, ffn_size)
                        # print(grad.size())
                        grad = grad.sum(dim=0)  # (ffn_size)
                        ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)  # (ffn_size)
                    ig_gold = ig_gold * weights_step  # (ffn_size)
                    res_dict['ig_gold'].append(ig_gold.tolist()) # (layer_num, ffn_size) # (12,3072)

                # sum integrated grad
                for i in range(len(res_dict['ig_gold'])):
                    for j in range(len(res_dict['ig_gold'][i])):
                        sum_list[i][j] += res_dict['ig_gold'][i][j]
   
            res_dict = convert_to_triplet_ig(sum_list)
            # res_dict = convert_to_triplet_ig(res_dict['ig_gold'])
            res_dict_bag.append([tokens_info, res_dict])

            fw.write(res_dict_bag)
            count += 1
            if count%100 == 0:
                print('Has processed ',count)
        # record running time
        toc = time.perf_counter()
        print(f"***** Private texts have been processed. Costing time: {toc - tic:0.4f} seconds *****")



if __name__ == "__main__":
    main()
