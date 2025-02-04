import pandas as pd
import string
import argparse

parser = argparse.ArgumentParser(description='Select n rows from a CSV file.')
parser.add_argument('--data_path', type=str, default="./data/肯定语体系【Caritas】 - 场景需求表V2.csv", help='Path to the CSV file.')
parser.add_argument('-n', type=int, default=10, help='Number of rows to select.')

args = parser.parse_args()
k = args.n


pathes = [f'/home/acszy/2025/Affirmative/data/select_{k}_result_2.csv',f'/home/acszy/2025/Affirmative/data/select_{k}_result_3.csv']
valid = []
invalid = []
for path in pathes:
    df = pd.read_csv(path)
    df['word_count'] = df['自我肯定语'].apply(lambda x: len(x.translate(str.maketrans('', '', string.punctuation))))
    # print(df.head())
    # print(df['word_count'].value_counts())

    # count how many less than 20
    valid.append(df[df['word_count'] < 20]['word_count'].count())
    invalid.append(df['word_count'].count())
    print(f"{100*df[df['word_count'] < 20]['word_count'].count()/df['word_count'].count():.2f}% ({df[df['word_count'] < 20]['word_count'].count()}/{df['word_count'].count()})   - {path.split('/')[-1]}")
print(f"{100*sum(valid)/sum(invalid):.2f}% ({sum(valid)}/{sum(invalid)})  - ALL")


