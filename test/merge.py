import pandas as pd
from tqdm import tqdm  # Import tqdm for the progress bar

# Read the first file
df = pd.read_csv('../data/标注结果_batch_1.csv')

# Loop through the remaining files and concatenate them
for i in tqdm(range(58), desc="Merging files"):  # Add tqdm for progress bar
    df_tmp = pd.read_csv(f'../data/标注结果_batch_{2+i}.csv')
    df = pd.concat([df, df_tmp], ignore_index=True)

# Save the merged DataFrame to a new CSV file
df.to_csv('../data/merged_5800.csv', index=False)