import pandas as pd
import string

pathes = ['/home/acszy/2025/Affirmative/data/select_10_result_2.csv','/home/acszy/2025/Affirmative/data/select_10_result_3.csv']
for path in pathes:
    df = pd.read_csv(path)
    df['word_count'] = df['自我肯定语'].apply(lambda x: len(x.translate(str.maketrans('', '', string.punctuation))))
    # print(df.head())
    # print(df['word_count'].value_counts())

    # count how many less than 20
    print(f"{df[df['word_count'] < 20]['word_count'].count()}/{df['word_count'].count()}")
