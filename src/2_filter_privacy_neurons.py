import jsonlines, json
import numpy as np
from collections import Counter
import os
import numpy as np
import sys
import re

import matplotlib.pyplot as plt
import seaborn as sns



rlts_dir = sys.argv[1]
kn_dir = sys.argv[1]+'kn/'

threshold_ratio = 0.1  # filter the neurons whose attribution score is less than threshold_ratio * the maximum value.
mode_ratio_bag = 0.5  # Filter out neurons whose frequency is less than mode_ratio_bag in the same batch of text



def re_filter(metric_triplets, total_metrix, cnt_metrix):
    metric_max = -999
    for i in range(len(metric_triplets)):
        metric_max = max(metric_max, metric_triplets[i][2])
    metric_triplets = [triplet for triplet in metric_triplets if triplet[2] >= metric_max * threshold_ratio]
    total_metrix += metric_max
    cnt_metrix += 1
    return metric_triplets, total_metrix, cnt_metrix


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]


def parse_kn(pos_cnt, tot_num, mode_ratio, min_threshold=0):
    mode_threshold = tot_num * mode_ratio
    mode_threshold = max(mode_threshold, min_threshold)
    kn_bag = []
    for pos_str, cnt in pos_cnt.items():
        if cnt >= mode_threshold:
            kn_bag.append(pos_str2list(pos_str))
    return kn_bag


def analysis_file(filename, metric='ig_gold'):
    rel = filename.split('.')[0].split('-')[-1]
    print(f'===========> parsing important position in {rel}..., mode_ratio_bag={mode_ratio_bag}')

    rlts_bag_list = []
    with open(os.path.join(rlts_dir, filename), 'r') as fr:
        for rlts_bag in jsonlines.Reader(fr):
            rlts_bag_list.append(rlts_bag)

    ave_kn_num = 0
    total_metrix = 0 
    cnt_metrix = 0
    kn_bag_list = []
    # get imp pos by bag_ig
    for bag_idx, rlts_bag in enumerate(rlts_bag_list):
        pos_cnt_bag = Counter()
        for rlt in rlts_bag:
            res_dict = rlt[1]
            metric_triplets, total_metrix, cnt_metrix = re_filter(res_dict, total_metrix, cnt_metrix)
            for metric_triplet in metric_triplets:
                pos_cnt_bag.update([pos_list2str(metric_triplet[:2])])
        kn_bag = parse_kn(pos_cnt_bag, len(rlts_bag), 1)
        ave_kn_num += len(kn_bag)
        kn_bag_list.append(kn_bag)

    ave_kn_num /= len(rlts_bag_list)
    # print(total_metrix/cnt_metrix)

    # get imp pos by rel_ig
    pos_cnt_rel = Counter()
    for kn_bag in kn_bag_list:
        for kn in kn_bag:
            pos_cnt_rel.update([pos_list2str(kn)])
    kn_rel = parse_kn(pos_cnt_rel, len(kn_bag_list), mode_ratio_bag)
    # print(len(kn_bag_list))
    # print(len(kn_rel))

    return ave_kn_num, kn_bag_list, kn_rel


def stat(data, pos_type, rel):
    if pos_type == 'kn_rel':
        print(f'{rel}\'s {pos_type} has {len(data)} imp pos. ')
        return
    ave_len = 0
    for kn_bag in data:
        ave_len += len(kn_bag)
    ave_len /= len(data)
    print(f'{rel}\'s {pos_type} has on average {ave_len} imp pos. ')



if not os.path.exists(kn_dir):
    os.makedirs(kn_dir)
for filename in os.listdir(rlts_dir):
    if filename.endswith('.priv.jsonl'):
        for max_it in range(4):
            ave_kn_num, kn_bag_list, kn_rel = analysis_file(filename)
            if ave_kn_num < 2:
                mode_ratio_bag -= 0.05
            if ave_kn_num > 10:
                mode_ratio_bag += 0.05
            if ave_kn_num >= 2 and ave_kn_num <= 10:
                break
        rel = filename.split('.')[0]
        stat(kn_bag_list, 'kn_bag', rel)
        stat(kn_rel, 'kn_rel', rel)
        with open(os.path.join(kn_dir, f'kn_bag-{rel}.json'), 'w') as fw:
            json.dump(kn_bag_list, fw, indent=2)
        with open(os.path.join(kn_dir, f'kn_rel-{rel}.json'), 'w') as fw:
            json.dump(kn_rel, fw, indent=2)


