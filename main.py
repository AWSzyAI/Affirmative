from src.utils import generate_self_affirmative_phrase_concurrent

def main():
    symptoms_file = "./data/select_100.csv"
    # symptoms_file = "./data/肯定语体系【Caritas】 - 场景-症状-需求.csv"
    output_file = symptoms_file.replace('.csv','_result.csv')
    checkpoint_file =  output_file.replace('_result.csv','_checkpoint.txt')
    delay=0.5
    max_retries=5

    generate_self_affirmative_phrase_concurrent(symptoms_file, output_file, checkpoint_file, n=5,DEBUG=True,delay=delay,max_retries=max_retries)

if __name__ == "__main__":
    main()

