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

def save_tels(data_path,Tel_list):
    Tel_list = list(set(Tel_list))
    with open(data_path,'w') as f:
        for i in Tel_list:
            final = i.replace('.','')+'\\n'
            f.write(final)

def add_prompted_nums(numbers):
    # process private numbers into fixed formula
    prompts = ['My phone number is ***. ','You can call me at ***. ', 'Dial *** to call me. ', 'Contact me at ***. ', 'Tel : ***. ', \
               'My fax number is ***. ', 'Fax : ***. ', 'You can send a fax to ***. ', 'I can receive faxes at ***. ', '*** is the fax number to reach me. ']

    tmp = prompts[random.randint(0,8)]
    return tmp.replace('***', numbers)


if __name__ == '__main__':
    # load data from /maildir/
    filelist = []
    for home, dirs, files in os.walk('/data/wuxinwei/dataset/maildir'): #/zufferli-j
        for filename in files:
            if '.' in filename:
                filelist.append(os.path.join(home, filename))

    str_list = []
    Tel_list = []

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
            elif 'MessageID' in lines[i]:
                pass
            elif 'Date: ' in lines[i]:
                pass
            elif 'From: ' in lines[i]:
                pass
            elif 'Subject: ' in lines[i]:
                pass
            elif 'Mime-Version: ' in lines[i]:
                pass
            elif 'Content-Type: ' in lines[i]:
                pass
            elif 'Content-Transfer-Encoding: ' in lines[i]:
                pass
            elif 'X-cc: ' in lines[i]:
                pass
            elif 'X-bcc: ' in lines[i]:
                pass
            elif 'X-Origin' in lines[i]:
                pass
            elif 'X-FileName: ' in lines[i]:
                pass
            else:
                temp_line = lines[i].strip() \
                    .replace('\\n',' ').replace('-','').replace('*','') \
                        .replace('#','').replace('_','').replace('=','')+' '
                number = ''.join(re.findall('\d+',temp_line))
                # processing numbers for easily encoding
                if len(number) == 10:               
                    temp_line = add_prompted_nums(' '.join(list(number)))
                    Tel_list.append(temp_line)

                temp_line = re.sub(' +', ' ', temp_line)
                temp_line = re.sub('\t+', ' ', temp_line)
                for j in refilter_email(temp_line):
                    # filter CC email addresses
                    temp_line = temp_line.replace(j+',','')
                new_line += (temp_line + ' ')
        if len(new_line.split(' ')) < 5:
            pass
        else:
            str_list.append(new_line.lstrip().lstrip('\t') +'\\n')
    
    
    # save_tels('./all_Tel.txt',Tel_list)
    split_and_save_data(str_list, "./temp_data/") 

