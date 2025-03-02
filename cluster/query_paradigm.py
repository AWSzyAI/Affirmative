# BUG 向量数据库检索有问题，所有的范式内容经过检索后返回的都是
# 情绪或者计算机网络通信这两篇文章。句子/文章的相似性评估存在问题。
# 

import sys
# 设置路径
sys.path.append('../src')
from utils import make_Affirmative_by_need, make_Affirmative, matched_paradigms,query_article,get_structured_articles,make_data_item,save_to_csv,HEADERS_structured_article,BAN_WORDS,HEADERS,logging
from prompt import get_role_prompt
from ark_api import MODEL_NAME
import markdown
import json

def md_to_json(md_file, json_file):
    # 读取Markdown文件内容
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 分行处理，逐个段落解析
    lines = md_content.split('\n')
    
    paradigms_data = []
    current_paradigm = {}
    
    for line in lines:
        # 检查是否是新的范式定义
        if line.startswith("## "):  # 检测新的范式名称
            if current_paradigm:
                paradigms_data.append(current_paradigm)  # 如果之前的范式已收集，保存它
            current_paradigm = {"范式名称": line[3:].strip(), "范式定义": "", "设计原则": "", "作用": "", "例句与解析": ""}
        
        # 检查范式的各个部分
        elif line.startswith("### 范式定义"):
            current_paradigm["范式定义"] = get_next_section(lines, lines.index(line))
        elif line.startswith("### 设计原则"):
            current_paradigm["设计原则"] = get_next_section(lines, lines.index(line))
        elif line.startswith("### 作用"):
            current_paradigm["作用"] = get_next_section(lines, lines.index(line))
        elif line.startswith("### 例句与解析"):
            current_paradigm["例句与解析"] = get_next_section(lines, lines.index(line))
    
    # 处理完所有行后，保存最后一个范式
    if current_paradigm:
        paradigms_data.append(current_paradigm)
    
    # 将数据写入到JSON文件
    with open(json_file, 'w', encoding='utf-8') as json_f:
        json.dump(paradigms_data, json_f, ensure_ascii=False, indent=4)

# 辅助函数：获取下一部分内容直到下一个 `###` 或结束
def get_next_section(lines, start_index):
    section_content = []
    for line in lines[start_index + 1:]:
        if line.startswith("### "):  # 下一个部分开始，停止提取
            break
        section_content.append(line.strip())
    return "\n".join(section_content).strip()

def get_paradigm_prompt(paradigm, json_file):
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        paradigms_data = json.load(f)
    
    # 遍历所有范式，查找与给定 paradigm 匹配的条目
    for entry in paradigms_data:
        if entry["范式名称"] == paradigm:
            return entry  # 返回找到的范式条目
    
    # 如果没有找到匹配的范式，返回一个空字典或适当的默认值
    return {}


def main():
    csv_file = "../data/query_paradigm.csv"
    paradigm_md_path = "../data/paradigm.md"
    md_file = '../data/paradigm.md'
    json_file = '../data/paradigm.json'
    # md_to_json(md_file, json_file)
    # print(f"Markdown文件已转换为JSON并保存到 {json_file}")


    for paradigm in matched_paradigms:
        paradigm_prompt = get_paradigm_prompt(paradigm, json_file)
        # {paradigm_prompt.get("例句与解析", "未找到")}
        query_message = f"""
            猫
        """
        # print
        # print(query_message)
        # save as log file
        with open("../data/query_message.log", "a") as f:
            f.write(query_message)
        article_data = query_article(query_message, 5)
        print(article_data)
        with open("../data/query_message.log", "a") as f:
            f.write(str(article_data))
        # return
        # continue
        zhihu_link = ' '.join([article['entity']['zhihu_link'] for article in article_data]) if article_data else "无链接"
        articles = ' '.join([article['entity']['content'] for article in article_data])
        structured_articles = get_structured_articles(article_data,"article-structurer")
        symptom = {
            "场景": '',
            "子场景": '',
            '用户需求': '',
            '场景描述': '',
            '心理作用机制与功能': '',
            '句子级别': '',
            '句子范式': ''
        }
        # 

        
        sentences = []
        messages = []
        role_maker = "Affirmative_maker-0213"
        for i, structured_article in enumerate(structured_articles):
            for j in ['状态描述：成为这样的我','发问：思考、反省','价值观','行动：可效仿的行动指南','慈悲：理解、接受、宽恕']:
                if structured_article.get(j):
                    affirmative,messages = make_Affirmative(role_maker,symptom,structured_article.get(j),articles=articles,messages=messages)
                    sentences.extend(affirmative)
            structured_item = make_data_item(
                type='structured_article', structured_articles=structured_article,
                symptom=symptom,
                )
            save_to_csv(csv_file.replace('.csv','_structured.csv'), structured_item, HEADERS_structured_article)
        sentences = list(set(sentences))
        if len(sentences) == 0:
            logging.warning(f"No sentences generated for symptom index {i}")
        else:
            sentences = list(set(sentences))
            for sentence in sentences:
                if any(word in sentence for word in BAN_WORDS):
                    continue
                # debug(sentence)
                type="0203-2"
                data_item = make_data_item(
                    self_affirmative_phrase=sentence,
                    type=type,
                    symptom=symptom,
                    zhihu_link=zhihu_link,
                    role = role_maker,
                    model = MODEL_NAME
                    )
                save_to_csv(csv_file.replace('.csv','_2.csv'), data_item,HEADERS)

        
        make_Affirmative_by_need(
            symptom, 
            paradigm, 
            sentences, 
            zhihu_link, 
            csv_file, 
            paradigm_md_path,
            messages=messages)
    
        
if __name__ == "__main__":
    main()    
