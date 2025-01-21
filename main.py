import sys
from src.utils import load_csv, generate_self_affirmative_phrase_concurrent

def main():
    # 检查命令行参数是否提供了自定义的句子
    if len(sys.argv) > 1:
        # 用户提供了单条数据，处理该数据
        user_sentence = sys.argv[1]
        additional_info = "默认附加信息"  # 可以根据需要修改为更合适的信息
        # 创建一个单独的症状数据条目
        symptoms_data = [{'用户问题/症状': user_sentence, '子场景症状合并': '', '标签（附加参考，用于引导生成或校正句子内容）': additional_info}]
        symptoms_file = "./data/single_test.csv"
    else:
        # symptoms_file = "./data/肯定语体系【Caritas】 - 场景-症状.csv"
        symptoms_file = "./data/select_1.csv"
        symptoms_data = load_csv(symptoms_file)
    output_file = symptoms_file.replace('.csv','_result.csv')
    checkpoint_file =  output_file.replace('_result.csv','_checkpoint.txt')
    generate_self_affirmative_phrase_concurrent(symptoms_data, output_file, checkpoint_file, n=5,DEBUG=True)

if __name__ == "__main__":
    main()
