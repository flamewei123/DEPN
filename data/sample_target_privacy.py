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


# data_path = './memorized_RANDOM.txt'   # './memorized_TEL.txt' './memorized_NAME.txt'
# target_path = './sampled_RANDOM.txt'  
# num = 100  

random_lines_to_file('./memorized_RANDOM.txt', './sampled_RANDOM.txt', 10)
random_lines_to_file('./memorized_TEL.txt', './sampled_TEL.txt', 20)
random_lines_to_file('./memorized_NAME.txt', 'sampled_NAME.txt', 20)
