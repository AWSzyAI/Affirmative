import pandas as pd
import string
import argparse
import progressbar
import sys
import time


def print_progress_bar(progress, total, bar_length=50):
    # Calculate the percentage
    percent = (progress / total) * 100
    bar = '#' * int(percent / 2) + ' ' * (bar_length - int(percent / 2))
    sys.stdout.write(f"\r{percent:6.2f}% ({progress}/{total}) [{'#' * int(percent / 2)}{' ' * (bar_length - int(percent / 2))}]")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description='Select n rows from a CSV file.')
    parser.add_argument('--data_path', type=str, default="./data/肯定语体系【Caritas】 - 场景需求表V2.csv", help='Path to the CSV file.')
    parser.add_argument('-n', type=int, default=10, help='Number of rows to select.')
    parser.add_argument('-m', '--max_length', type=int, default=20, help='Maximum length of the affirmative phrase.')

    args = parser.parse_args()
    k = args.n
    max_length = args.max_length

    pathes = [f'/home/acszy/2025/Affirmative/data/select_{k}_result_2.csv', f'/home/acszy/2025/Affirmative/data/select_{k}_result_3.csv']
    valid = []
    invalid = []
    total_files = len(pathes)
    for idx, path in enumerate(pathes):
        df = pd.read_csv(path)
        df['word_count'] = df['自我肯定语'].apply(lambda x: len(x.translate(str.maketrans('', '', string.punctuation))))

        valid_count = df[df['word_count'] < max_length]['word_count'].count()
        invalid_count = df['word_count'].count()

        valid.append(valid_count)
        invalid.append(invalid_count)

        # Print progress for each file
        print_progress_bar(valid_count, invalid_count)

        print(f" - {path.split('/')[-1]}")

    # Final print for all
    total_valid = sum(valid)
    total_invalid = sum(invalid)
    total_percentage = 100 * total_valid / total_invalid

    # Final progress bar for all files
    print_progress_bar(total_valid, total_invalid)

    # Print the final result
    print(f" - ALL")

if __name__ == "__main__":
    main()