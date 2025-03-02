import json
import csv

def md_to_json(md_file, json_file):
    # 读取Markdown文件内容
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 分行处理，逐个段落解析
    lines = md_content.split('\n')
    
    paradigms_data = []
    current_paradigm = {}
    current_section = None
    
    for line in lines:
        # 检查是否是新的范式定义
        if line.startswith("## "):  # 检测新的范式名称
            if current_paradigm:
                paradigms_data.append(current_paradigm)  # 如果之前的范式已收集，保存它
            current_paradigm = {"范式名称": line[3:].strip(), "范式定义": [], "设计原则": [], "作用": [], "例句与解析": []}
            current_section = None  # 重置当前部分
        
        # 检查范式的各个部分
        elif line.startswith("### 范式定义"):
            current_section = "范式定义"
        elif line.startswith("### 设计原则"):
            current_section = "设计原则"
        elif line.startswith("### 作用"):
            current_section = "作用"
        elif line.startswith("### 例句与解析"):
            current_section = "例句与解析"
        
        # 如果当前行不是标题，且当前部分已定义，则将内容添加到当前部分
        elif current_section:
            if line.strip():  # 如果行不为空
                current_paradigm[current_section].append(line.strip())
    
    # 处理完所有行后，保存最后一个范式
    if current_paradigm:
        paradigms_data.append(current_paradigm)
    
    # 将数据写入到JSON文件
    with open(json_file, 'w', encoding='utf-8') as json_f:
        json.dump(paradigms_data, json_f, ensure_ascii=False, indent=4)

    print(f"数据已成功写入到 {json_file}")




md_file = '../data/paradigm.md'
json_file = '../data/paradigm.json'
csv_file = "../data/paradigm_energy_levels.csv"  # 输出的 CSV 文件路径

# md_to_json(md_file, json_file)



import re

def extract_paradigm_energy_levels(json_file, csv_file):
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        paradigms_data = json.load(f)
    
    # 准备 CSV 文件
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["范式名称", "范式能量层级", "能量层级编号"])  # 写入表头

        # 遍历每个范式
        for paradigm in paradigms_data:
            paradigm_name = paradigm["范式名称"]
            definition_lines = paradigm["范式定义"]
            
            # 初始化变量
            energy_level_full = "未定义"
            energy_level_code = "未定义"
            
            # 查找范式能量层级
            for line in definition_lines:
                if "范式能量层级" in line:
                    energy_level_full = line.split("：")[1].strip()  # 提取完整的范式能量层级描述
                    # 使用正则表达式提取层级编号（如 L1、L2）
                    match = re.search(r"L\d+", energy_level_full)
                    if match:
                        energy_level_code = match.group()  # 提取匹配到的层级编号
                    break
            
            # 写入 CSV 文件
            csv_writer.writerow([paradigm_name, energy_level_full, energy_level_code])

    print(f"数据已成功写入到 {csv_file}")

# 示例调用

extract_paradigm_energy_levels(json_file, csv_file)