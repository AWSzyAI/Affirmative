import argparse
from src.utils import generate_self_affirmative_phrase_concurrent
from datetime import datetime
import os
import shutil

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
    paradigm_md_path = "./data/paradigm.md"

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
        paradigm_md_path,
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