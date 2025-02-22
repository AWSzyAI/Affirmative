# import argparse
# import json
# from src.utils import generate_self_affirmative_phrase_concurrent
# from datetime import datetime
# import os
# import shutil

# def load_config(config_path='config.json'):
#     """加载JSON配置文件"""
#     with open(config_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def main():
#     # 加载配置文件
#     config = load_config()
#     file_paths = config['file_paths']
#     function_params = config['function_params']
    
#     # 解析命令行参数
#     parser = argparse.ArgumentParser(description="Generate self-affirmative phrases from a CSV file.")
#     parser.add_argument('-n', type=int, required=False, help="The value of k to select the CSV file.")
#     parser.add_argument('-m', '--max_length', type=int, required=True, help="max_length of sentence")
#     args = parser.parse_args()

#     # 确定输入文件路径
#     if args.n is not None:
#         input_file = os.path.join(
#             file_paths['input_data_dir'],
#             file_paths['input_data_template'].format(n=args.n)
#         )
#     else:
#         input_file = os.path.join(
#             file_paths['input_data_dir'],
#             file_paths['default_input_file']
#         )

#     # 生成时间戳文件夹
#     timestamp_folder = os.path.join(
#         file_paths['output_base_dir'],
#         datetime.now().strftime(config['timestamp_format'])
#     )
#     os.makedirs(timestamp_folder, exist_ok=True)

#     # 构造输出路径
#     output_file = os.path.join(
#         timestamp_folder,
#         os.path.basename(input_file).replace('.csv', '_result.csv')
#     )
#     checkpoint_file = output_file.replace('_result.csv', '_checkpoint.txt')
#     log_file = output_file.replace('_result.csv', '.log')

#     # 备份输入文件
#     if config.get('backup_input_file', False):
#         shutil.copy(input_file, timestamp_folder)

#     print(f"Input File: {input_file}")
#     print(f"Output File: {output_file}")
#     print(f"Checkpoint File: {checkpoint_file}")

#     # 调用生成函数
#     generate_self_affirmative_phrase_concurrent(
#         symptoms_file=input_file,
#         output_file=output_file,
#         checkpoint_file=checkpoint_file,
#         n=function_params['n'],
#         DEBUG=function_params['debug'],
#         delay=function_params['delay'],
#         max_retries=function_params['max_retries'],
#         use_concurrency=function_params['use_concurrency'],
#         max_length=args.max_length,
#         timeout=function_params['timeout'],
#         log_file=log_file
#     )

# if __name__ == "__main__":
#     main()
    
import argparse
from src.utils import generate_self_affirmative_phrase_concurrent
from datetime import datetime
import os
import shutil

# def main():
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser(description="Generate self-affirmative phrases from a CSV file.")
#     parser.add_argument('-n', type=int, required=False, help="The value of k to select the CSV file.")
#     parser.add_argument('-m','--max_length', type=int, required=True, help="max_length of sentence")
#     args = parser.parse_args()

#     # 检查最近的timestamp_folder中是否有checkpoint文件，
#     # 如果有，直接读取checkpoint文件，继续在当前的timestamp_folder中继续生成
#     # 如果没有，常见新的timestamp_folder，重新生成
    

#     # 生成时间戳文件夹
#     timestamp_folder = "./output/" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#     os.makedirs(timestamp_folder, exist_ok=True)  # 确保目录存在

#     # symptoms_file = "./data/肯定语体系【Caritas】 - 场景-症状-需求_去重后.csv"
#     symptoms_file = f"./data/select_{args.n}.csv"
#     # output_file = symptoms_file.replace('.csv', '_result.csv')
#     # checkpoint_file = output_file.replace('_result.csv', '_checkpoint.txt')
#     output_file = os.path.join(timestamp_folder, os.path.basename(symptoms_file).replace('.csv', '_result.csv'))
#     checkpoint_file = output_file.replace('_result.csv', '_checkpoint.txt')
#     log_file = output_file.replace('_result.csv', '.log') 
#     # 拷贝一份文件到timestamp_folder
    
#     shutil.copy(symptoms_file, timestamp_folder)

#     print(f"Output File: {output_file}")
#     print(f"Checkpoint File: {checkpoint_file}")

    
#     # # Parameters for the function
#     delay = 0.5
#     max_retries = 5

#     # Call the function
#     generate_self_affirmative_phrase_concurrent(
#         symptoms_file, 
#         output_file, 
#         checkpoint_file, 
#         n=5, 
#         # DEBUG=False, 
#         DEBUG=True, 
#         delay=delay, 
#         max_retries=max_retries, 
#         # use_concurrency=False,
#         use_concurrency=True,
#         max_length=args.max_length,
#         timeout=1800,  # 设置30分钟超时
#         log_file = log_file
#     )

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Generate self-affirmative phrases from a CSV file.")
    parser.add_argument('-n', type=int, required=False, help="The value of k to select the CSV file.")
    parser.add_argument('-m', '--max_length', type=int, required=True, help="max_length of sentence")
    args = parser.parse_args()

    # 确定症状文件路径和检查点文件名
    symptoms_file = f"./data/select_{args.n}.csv" if args.n else "./data/default_symptoms.csv"
    symptoms_file_basename = os.path.basename(symptoms_file)
    checkpoint_filename = symptoms_file_basename.replace('.csv', '_checkpoint.txt')

    # 检查输出目录是否存在并处理检查点
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    valid_subdirs = []

    # 遍历所有子目录查找匹配的检查点文件
    for d in os.listdir(output_dir):
        dir_path = os.path.join(output_dir, d)
        if os.path.isdir(dir_path):
            cp_path = os.path.join(dir_path, checkpoint_filename)
            if os.path.isfile(cp_path):
                valid_subdirs.append(d)

    # 按时间戳排序子目录
    valid_subdirs.sort(key=lambda x: datetime.strptime(x, '%Y-%m-%d_%H-%M-%S'), reverse=True)

    # 确定时间戳文件夹路径
    if valid_subdirs:
        timestamp_folder = os.path.join(output_dir, valid_subdirs[0])
        print(f"Resuming from existing checkpoint in {timestamp_folder}")
    else:
        timestamp_folder = os.path.join(output_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(timestamp_folder, exist_ok=True)
        shutil.copy(symptoms_file, timestamp_folder)
        print(f"Created new output directory {timestamp_folder}")

    
    # 生成输出文件路径
    output_file = os.path.join(timestamp_folder, symptoms_file_basename.replace('.csv', '_result.csv'))
    checkpoint_file = os.path.join(timestamp_folder, checkpoint_filename)
    log_file = os.path.join(timestamp_folder, symptoms_file_basename.replace('.csv', '.log'))

    print(f"Output File: {output_file}")
    print(f"Checkpoint File: {checkpoint_file}")
    
    # return
    # 其余原有代码保持不变
    delay = 0.5
    max_retries = 5

    generate_self_affirmative_phrase_concurrent(
        symptoms_file, 
        output_file, 
        checkpoint_file, 
        n=5, 
        DEBUG_model=True, 
        delay=delay, 
        max_retries=max_retries,
        use_concurrency=True,
        max_length=args.max_length,
        timeout=1800,
        log_file=log_file
    )

if __name__ == "__main__":
    main()
    


# from src.utils import generate_self_affirmative_phrase_concurrent

# def main():
#     # 接受参数k： -k 10
#     symptoms_file = f"./data/select_{k}.csv"
#     # symptoms_file = "./data/select_1.csv"
#     # symptoms_file = "./data/肯定语体系【Caritas】 - 场景-症状-需求_去重后.csv"
#     output_file = symptoms_file.replace('.csv','_result.csv')
#     checkpoint_file =  output_file.replace('_result.csv','_checkpoint.txt')
#     delay=0.5
#     max_retries=5

#     generate_self_affirmative_phrase_concurrent(symptoms_file, output_file, checkpoint_file, n=5,DEBUG=True,delay=delay,max_retries=max_retries,use_concurrency=False)

# if __name__ == "__main__":
#     main()

