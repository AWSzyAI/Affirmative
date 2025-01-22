import pandas as pd
import random
import os
import argparse

HEADERS = ['用户1级需求','用户2级需求']

def select_n(data_path, n):
    data = pd.read_csv(data_path)
    data = data[HEADERS]
    data = data.sample(n)
    data.to_csv(f"./data/select_{n}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select n rows from a CSV file.')
    # parser.add_argument('--data_path', type=str, default="./data/肯定语体系【Caritas】 - 场景-症状.csv", help='Path to the CSV file.')
    parser.add_argument('--data_path', type=str, default="./data/肯定语体系【Caritas】 - 场景-症状-需求.csv", help='Path to the CSV file.')
    parser.add_argument('-n', type=int, default=10, help='Number of rows to select.')
    
    args = parser.parse_args()
    
    select_n(args.data_path, args.n)
