
import os
import glob
import argparse

parser = argparse.ArgumentParser(description='Select n rows from a CSV file.')
parser.add_argument('--data_path', type=str, default="./data/肯定语体系【Caritas】 - 场景需求表V2.csv", help='Path to the CSV file.')
parser.add_argument('-n', type=int, default=10, help='Number of rows to select.')

args = parser.parse_args()
k = args.n

PATHs = [f'/home/acszy/2025/Affirmative/data/select_{k}_result_*.csv']
for path in PATHs:
    for file_path in glob.glob(path):
        if os.path.exists(file_path):
            os.remove(file_path)