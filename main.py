import sys
from src.utils import load_csv, generate_self_affirmative_phrase_concurrent

def main():
    symptoms_file = "./data/select_10.csv"
    # symptoms_file = "./data/肯定语体系【Caritas】 - 场景-症状-需求.csv"
    symptoms_data = load_csv(symptoms_file)
    output_file = symptoms_file.replace('.csv','_result.csv')
    checkpoint_file =  output_file.replace('_result.csv','_checkpoint.txt')
    generate_self_affirmative_phrase_concurrent(symptoms_data, output_file, checkpoint_file, n=5,DEBUG=True)

if __name__ == "__main__":
    main()
