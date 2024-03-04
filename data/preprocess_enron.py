#-*- coding : utf-8-*-
# coding:unicode_escape
import os
from tqdm import tqdm
import re
import random
  
def split_and_save_data(str_list, enron_path):  
    if not os.path.exists(enron_path):  
        os.makedirs(enron_path)  
      
    train_path = os.path.join(enron_path, "train.txt")  
    test_path = os.path.join(enron_path, "test.txt")  
    valid_path = os.path.join(enron_path, "valid.txt")  
      
    with open(train_path, "w", encoding="utf-8") as train_file, \
         open(test_path, "w", encoding="utf-8") as test_file, \
         open(valid_path, "w", encoding="utf-8") as valid_file:  
          
        valid_data = str_list[-3000:]  
        for item in valid_data:  
            valid_file.write(item)  
          
        test_data = str_list[-6000:-3000]  
        for item in test_data:  
            test_file.write(item)  
          
        train_data = str_list[:-6000]  
        for item in train_data:  
            train_file.write(item)  

# def get_name(filename):
#     return 0

def refilter_email(text):
    emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text)
    return emails

def add_prompted_nums(numbers,typ):
    # process private numbers into fixed formula
    tel_p = ['My phone number is ***','You can call me at ***', 'Dial *** to call me', 'Contact me at ***', 'Tel : ***']
    fax_p = ['My fax number is ***', 'Fax : ***', 'You can send a fax to ***', 'I can receive faxes at ***', '*** is the fax number to reach me']
    if typ == 'Tel':
        n = random.randint(0,4)
        tmp = tel_p[n]
        return tmp.replace('***', numbers)
    if typ == 'Fax':
        n = random.randint(0,4)
        tmp = fax_p[n]
        return tmp.replace('***', numbers)

if __name__ == '__main__':
    # load data from /maildir/
    filelist = []
    for home, dirs, files in os.walk('/data/wuxinwei/dataset/maildir'): #/zufferli-j
        for filename in files:
            if '.' in filename:
                filelist.append(os.path.join(home, filename))

    str_list = []

    for file in tqdm(filelist):
        with open(file, 'r') as f:
            try:
                lines = f.readlines()
            except Exception as e:
                pass
        new_line = ''
        for i in range(len(lines)):
            if "X-FileName:" in lines[i]:
                pass
            temp_line = lines[i].strip().replace('ContentTransferEncoding: 7bit','') \
                .replace('\\n',' ').replace('-','').replace('*','') \
                    .replace('#','').replace('_','').replace('=','')+' '
            numbers = re.findall(r'\b\d{10}\b',lines[i])
            # processing numbers for easily encoding
            if numbers:
                for num in numbers:
                    temp_line = temp_line.replace(num, ' '.join(list(num)))                

            temp_line = re.sub(' +', ' ', temp_line)
            for j in refilter_email(temp_line):
                # filter CC email addresses
                temp_line = temp_line.replace(j+',','')
            new_line += (temp_line + ' ')
        if len(new_line.split(' ')) < 5:
            pass
        else:
            str_list.append(new_line.lstrip().lstrip('\t'))   # +'\\n')

    # for i in range(10):
    #     print(str_list[i])
    split_and_save_data(str_list, "./enron_data") 
