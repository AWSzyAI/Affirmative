import argparse
from src.utils import generate_self_affirmative_phrase_concurrent

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate self-affirmative phrases from a CSV file.")
    parser.add_argument('-n', type=int, required=False, help="The value of k to select the CSV file.")
    parser.add_argument('-m','--max_length', type=int, required=True, help="max_length of sentence")
    args = parser.parse_args()

    # Construct file paths
    # symptoms_file = "./data/select_1.csv"
    # symptoms_file = "./data/肯定语体系【Caritas】 - 场景-症状-需求_去重后.csv"
    symptoms_file = f"./data/select_{args.n}.csv"
    output_file = symptoms_file.replace('.csv', '_result.csv')
    checkpoint_file = output_file.replace('_result.csv', '_checkpoint.txt')

    # Parameters for the function
    delay = 0.5
    max_retries = 5

    # Call the function
    generate_self_affirmative_phrase_concurrent(
        symptoms_file, 
        output_file, 
        checkpoint_file, 
        n=5, 
        DEBUG=True, 
        delay=delay, 
        max_retries=max_retries, 
        # use_concurrency=False,
        use_concurrency=True,
        max_length=args.max_length,
        timeout=1800  # 设置30分钟超时
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

