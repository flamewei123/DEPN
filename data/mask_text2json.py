import json
import re
from transformers import AutoTokenizer

def get_masked_text(line,txt_type):
    if txt_type == 'TEL':
        text = line.split('#')[0].strip()
        tel_numbers = re.findall(r'\d(?:\s\d){9}', text)[0].split(' ')
        prompt = text.replace(' '.join(tel_numbers),'***')
        res = []
        for i in range(len(tel_numbers)):
            data = []
            temp = re.findall(r'\d(?:\s\d){9}', text)[0].split(' ')
            gl = temp[i]
            temp[i] = '[MASK]'
            new_num = ' '.join(temp)
            data.append(prompt.replace('***',new_num))
            data.append(gl)
            res.append(data)
    if txt_type == 'NAME':
        text = line.split('#')[0].strip()
        name = line.split('#')[1].strip().split(' ')
        res = []
        for i in range(len(name)):
            data = []
            tokens = text.split(' ')
            tokens[tokens.index(name[i])] = '[MASK]'
            gl = name[i]
            data.append(' '.join(tokens))
            data.append(gl)
            res.append(data)
    if txt_type == 'RANDOM':
        text = line.split('#')[0].strip()
        tokenizer = AutoTokenizer.from_pretrained('/data/wuxinwei/transformers/model/2024/0305_ep10_bert_base')
        tokenized_text = tokenizer.tokenize(text)
        mask_token = tokenizer.mask_token
        res = []

        for i in range(len(tokenized_text)):
            data = []
            temp_text = tokenized_text.copy()
            gl = temp_text[i]
            temp_text[i] = mask_token
            masked_sentence = ' '.join(temp_text).replace(' ##', '')
            data.append(masked_sentence)
            data.append(gl)
            res.append(data)

    return res

def text2json(prvacy_kind):
    with open('sampled_'+prvacy_kind+'.txt','r') as f:
        texts = f.readlines()
        output = []
        for text in texts:
            data_list = get_masked_text(text.strip(),prvacy_kind)
            output.append(data_list)
        with open('sampled_'+prvacy_kind+'.json','w') as f:
            json.dump(output, f, indent=2)
    f.close()

text2json('NAME')
text2json('TEL')
text2json('RANDOM')
