import pandas as pd
import sys
import logging
import coloredlogs
from tqdm import tqdm
import concurrent.futures
import os
import json
from threading import Lock,Thread

sys.path.append('../src')
from prompt import get_role_prompt
from kimi_api import client, MODEL_NAME

# 配置日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建文件处理器并设置日志文件路径
log_file = 'app.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# 创建控制台输出处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器到logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
coloredlogs.install(level='INFO', logger=logger)

checkpoint_lock = Lock()

def get_content_by_type(row, type):
    if type == '合集':
        message = f"""
        请根据以下“自我肯定语”内容，判断其适合推送给哪类用户，并返回推荐标签。

        自我肯定语: {row['自我肯定语']}

        ### 背景说明：

        我们将所有合集按照下列类别划分，每类代表用户的一种心理需求：

        1. **疗愈类（40%）**：平静内心、抗击抑郁、感恩练习、减压放松、心碎疗愈、艰难时刻
           - 特征：L1~L2能量级，关注心理支持、恢复与安抚。
           - 标签推荐：情感疗愈、治愈之旅、平和心境、自我关怀
        
        2. **成长/激励类（60%）**：
           - 自我关怀：爱我的身体、爱自己、自我关怀、自尊心
           - 积极思考：积极心态、愉悦心情、焕发生命力、点燃动力
           - 自信：自信
           - 女性主义：拥抱女性身份、女性力量
           - 人际关系：亲密关系、人际交往、爱家人、原谅
           - 个人成长：个人成长、事业成功
           - 成为英雄：擦拭信念、追梦无悔、勇气
           - 特征：L3~L6能量级，更具激励性、行动导向。
           - 标签推荐：自信、个人成长、人际交往、成为英雄

合集名称,类别
平静内心,疗愈类
抗击抑郁,疗愈类
感恩练习,疗愈类
减压放松,疗愈类
心碎疗愈,疗愈类
艰难时刻,疗愈类
爱我的身体,自我关怀
爱自己,自我关怀
自我关怀,自我关怀
自尊心,自我关怀
积极心态,积极思考
愉悦心情,积极思考
焕发生命力,积极思考
点燃动力,积极思考
自信,自信
拥抱女性身份,女性主义
女性力量,女性主义
亲密关系,人际关系
人际交往,人际关系
爱家人,人际关系
原谅,人际关系
个人成长,个人成长
事业成功,个人成长
擦拭信念,成为英雄
追梦无悔,成为英雄
勇气,成为英雄


        ### 任务要求：

        1. 判断该自我肯定语更适合属于哪个合集类别（可多选）。
        2. 按照图中合集名称与类别的对应关系，从以下标签中选择适配的内容：
           - 情感疗愈、治愈之旅、平和心境、自我关怀（适合疗愈类合集）
           - 自信、个人成长、成为英雄、人际交往（适合成长/激励类合集）
        3. 结合语句的能量级别（L1~L6）判断是否适合标注某些标签，避免“高能量”语句被错误标注为“情感疗愈”等。
        4. 返回 JSON 格式结果，包括：
            - "key": 固定值 "合集"
            - "value": 合适的标签列表
            - "note": 简要说明未选某些标签的理由（例如能量级不符）

        ### 示例返回格式：
        ```json
        {{
            "key": "合集",
            "value": ["个人成长", "自信"],
            "note": "未选‘情感疗愈’和‘治愈之旅’，因该语句强调目标与成就，情感支持不明显。"
        }}
        ```
        """
    elif type == '感情状况':
        message = f"""
        根据以下自我肯定语及其相关信息，判断是否适合推送给处于特定感情状况的用户，并根据适合的感情状况返回JSON格式的结果。

        自我肯定语: {row['自我肯定语']}
    

        1. 判断该自我肯定语及其场景描述是否适合以下感情状况的用户：
        ["正在恋爱", "最近刚分手", "处在一段艰难的亲密关系中", "快乐地单身着", "单身但准备好了开始新的恋情", "有点辛苦的暗恋"]。
        2. 选择一个或多个感情状况标签进行标注，**不要**选择其他标签。
        3. 返回的标签列表应该准确反映与该自我肯定语的适配情况。
        4. 如果不选择某些标签，则在"note"中给出理由。选择的标签不用给出。

        请以如下格式返回结果：
        {{
            "key": "感情状况",
            "value": ["正在恋爱", "单身但准备好了开始新的恋情"],
            "note": "不放一些标签的理由"
        }}
        """
    elif type == '最近的感觉':
        message = f"""
        请根据以下自我肯定语及其相关信息，判断它们是否适合推送给具有特定情感状态的用户，并根据用户的情感状态返回相应的JSON格式结果。

        自我肯定语: {row['自我肯定语']}

        1. 判断该自我肯定语及其场景描述是否适合推送给以下情感状态的用户：
        ["开心", "很好", "一般", "不好", "糟糕"]。
        2. 请根据句子的情感倾向选择一个或多个标签：  
        - 【开心】和【很好】可以同时出现，  
        - 【不好】和【糟糕】也可以同时出现，  
        - 不要选择额外的标签。
        3. 1级句子倾向于【不好】和【糟糕】，2级句子倾向于【开心】和【很好】，但不是绝对的，具体分析。
        4. "什么让你有这种感觉"是对"最近的感觉"的解释，请从以下选项中选择与感受相关的因素：
        ["家庭", "朋友", "工作", "健康", "感情", "学业", "自己"]。
        5. 如果不选择某些标签，则在"note"中给出理由。选择的标签不用给出。

        请以如下格式返回结果：
        {{
            "最近的感觉": ["不好", "糟糕"],
            "什么让你有这种感觉": ["家庭", "健康"],
            "note": "不放一些标签的理由"
        }}
        """
    return message

def update_progress(pbar):
    """更新进度条"""
    pbar.update(1)

def get_checkpoint(checkpoint_file):
    """获取 checkpoint 文件中记录的所有已完成索引"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        return checkpoint_data
    else:
        # 如果 checkpoint 文件不存在，初始化为空列表
        with open(checkpoint_file, 'w') as f:
            json.dump([], f)
        return []

def update_checkpoint(checkpoint_file, index):
    """更新 checkpoint 文件（线程安全）"""
    with checkpoint_lock:  # 使用线程锁确保线程安全
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
        else:
            checkpoint_data = []
        
        if index not in checkpoint_data:
            checkpoint_data.append(index)

        # 写回检查点文件
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

def save_results(row_results, input_file):
    """保存处理结果到文件"""
    output_file = input_file.replace('.csv', '_result.csv')
    retry_output_file = input_file.replace('.csv', '_result_retry.csv')

    # 允许的值
    valid_collections = {"情感疗愈", "自信", "平和心境", "治愈之旅", "人际交往", "自我关怀", "个人成长", "成为英雄"}
    valid_relationships = {"正在恋爱", "最近刚分手", "处在一段艰难的亲密关系中", "快乐地单身着", "单身但准备好了开始新的恋情", "有点辛苦的暗恋"}
    valid_feelings = {"开心", "很好", "一般", "不好", "糟糕"}
    valid_causes = {"家庭", "朋友", "工作", "健康", "感情", "学业", "自己"}

    # 确保 "合集" 是合法子集
    collection_set = set(row_results.get("合集", []))  # "合集" 是一个列表
    if not collection_set.issubset(valid_collections):
        save_file = retry_output_file
    else:
        # 确保 "感情状况" 中的每个值都在允许的集合中
        relationship_list = row_results.get("感情状况", [])
        feeling_list = row_results.get("最近的感觉", [])
        cause_list = row_results.get("什么让你有这种感觉", [])

        relationship_valid = all(r in valid_relationships for r in relationship_list)
        feeling_valid = all(f in valid_feelings for f in feeling_list)
        cause_valid = all(c in valid_causes for c in cause_list)

        # 只要有一个不符合，就存入 retry 文件
        if not (relationship_valid and feeling_valid and cause_valid):
            save_file = retry_output_file
        else:
            save_file = output_file


    row_results_df = pd.DataFrame([row_results])
    row_results_df.to_csv(save_file, mode='a', header=not os.path.exists(output_file), index=False)

def process_row(index, row, checkpoint_file, input_file, pbar):
    """处理每一行数据并保存进度"""
    logger.debug(f"正在处理第 {index} 行数据")  # 调试日志
    row_results = {}

    # 将自我肯定语添加为字典项
    row_results.update({'自我肯定语': row['自我肯定语']})
    
    for type in ['合集', '感情状况', '最近的感觉']:
        annotation_result = get_annotation(row, type)
        row_results.update(annotation_result)
    
    # 更新其它列
    # row_results.update({'场景': row['场景']})
    # row_results.update({'子场景': row['子场景']})
    # row_results.update({'用户需求': row['用户需求']})
    # row_results.update({'心理作用机制与功能': row['心理作用机制与功能']})
    # row_results.update({'句子级别': row['句子级别']})

    # 存储处理结果到文件
    save_results(row_results, input_file)

    # 更新checkpoint并记录日志
    update_checkpoint(checkpoint_file, index)
    logger.info(f"已更新checkpoint，已处理行：{index}")
    
    # 更新进度条
    Thread(target=update_progress, args=(pbar,)).start()


def clean_value(value):
    """清理字段中的换行符"""
    if isinstance(value, str):
        return value.replace('\n', ' ').replace('\r', ' ')  # 删除换行符和回车符
    return value


def load_checkpoint(checkpoint_file):
    """从文件加载checkpoint，返回已处理的行数"""
    checkpoint_data = set(get_checkpoint(checkpoint_file)) 
    return checkpoint_data

def process_file(input_file):
    """主任务函数：处理文件中的每一行数据"""
    data = pd.read_csv(input_file)
    # data = data.sample(10)
    checkpoint_file = input_file.replace('.csv','_checkpoint.txt')
    output_file = input_file.replace('.csv','_result.csv')
    
    # 2025-02-06 工作：
    # 检查output_file是否存在，checkpoint是否存在
    # 如果output_file存在但checkpoint不存在，就
    # 比对 data与 output_file的自我肯定语,生产者,场景,子场景,用户需求,心理作用机制与功能,句子级别
    # 如果一模一样，就把 i 添加到初始化的checkpoint.txt中
    
    completed_indices = load_checkpoint(checkpoint_file)
    print(f"从检查点文件读取到已完成的索引: {completed_indices}")
    
    # 使用进度条显示处理进度
    with tqdm(total=len(data),initial=len(completed_indices),desc='进度',unit='task',position=0) as pbar:
        MAX_CONCURRENT_WORKERS = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
            futures = {}
            for index, row in data.iterrows():
                if index in completed_indices:
                    pbar.update(1)
                    continue
                future = executor.submit(
                    process_row, index, row, checkpoint_file, input_file, pbar
                )
                futures[future] = index
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    future.result()
                    Thread(target=update_progress,args=(pbar,)).start()
                    update_checkpoint(checkpoint_file, index)
                except Exception as e:
                    print(f"任务{index}失败: {e}")
                    raise


# def get_annotation(row, type='合集'):
#     role = f"noter-0205-{type}"
#     role_prompt = get_role_prompt(role)
#     context = get_content_by_type(row, type)
#     user_prompt = f"{role_prompt}\n{context}"
#     result = client.ask(MODEL_NAME, user_prompt)
#     result_json = json.loads(result['data']['text'])
#     return result_json

def get_annotation(row, type='合集'):
    role = f"noter-0205-{type}"
    role_prompt = get_role_prompt(role=role)
    
    # 根据不同类型生成提示词并返回相应的标注结果
    content = get_content_by_type(row, type)
    messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": content}
    ]
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=1,
            response_format={"type": "json_object"},  # 确保返回 JSON 格式
            n=1  # 请求返回1个结果
        )
        response = completion.choices[0].message.content.strip()
        response_dict = json.loads(response)
        
        result = {}
        if type == "合集":
            result['合集'] = response_dict.get('value', [])
            result['合集理由'] = clean_value(response_dict.get('note', []))
        elif type == "感情状况":
            result['感情状况'] = response_dict.get('value', [])
            result['感情状况理由'] = clean_value(response_dict.get('note', []))
        elif type == "最近的感觉":
            result['最近的感觉'] = response_dict.get("最近的感觉", [])
            result['什么让你有这种感觉'] = response_dict.get('什么让你有这种感觉', [])
            result['最近的感觉理由'] = clean_value(response_dict.get('note', []))
        
        logger.info(f"成功获取{type}标注")
        return result
    except Exception as e:
        logger.error(f"获取{type}标注时发生错误: {e}")
        raise
        return {}

if __name__ == "__main__":
    input_file = '../data/0315句子更新 - 汇总表_result_retry.csv'
    # input_file = '../data/test.csv'
    # checkpoint_file = input_file.replace('.csv','_checkpoint.txt')
    # process_file(input_file, checkpoint_file)
    process_file(input_file)


# import pandas as pd
# import sys
# import logging
# import coloredlogs  # 引入coloredlogs库
# from tqdm import tqdm  # 导入tqdm
# sys.path.append('../src')
# from prompt import get_role_prompt
# from kimi_api import client, MODEL_NAME
# import json

# # 配置日志
# logger = logging.getLogger()
# coloredlogs.install(level='INFO', logger=logger)

# def get_content_by_type(row, type):
#     if type == '合集':
#         message = f"""
#         请根据以下自我肯定语及其相关信息，判断它们是否适合推送给具有特定标签的用户，并根据适配的标签返回JSON格式的结果。

#         自我肯定语: {row['自我肯定语']}
#         场景: {row['场景']}
#         子场景: {row['子场景']}
#         用户需求: {row['用户需求']}
#         心理作用机制与功能: {row['心理作用机制与功能']}
#         句子级别: {row['句子级别']}
        
#         1. 判断这个自我肯定语是否适合推送给以下标签的用户：["情感疗愈", "自信", "平和心境", "治愈之旅", "人际交往", "自我关怀", "个人成长", "成为英雄"]。
#         2. 选择一个或多个标签进行标注，**不要**捏造标签，确保标签与自我肯定语的情感和语境相符。
#         3. 如果不选择某些标签，则在"note"中给出理由。选择的标签不用给出。

        
#         请返回符合条件的标签列表，并以标准JSON格式返回结果，如下所示：
#         {{
#             "key": "合集",
#             "value": ["情感疗愈", "自信", "治愈之旅"],
#             "note":"不放一些标签的理由"
#         }}
#         """
#     elif type == '感情状况':
#         message = f"""
#         根据以下自我肯定语及其相关信息，判断是否适合推送给处于特定感情状况的用户，并根据适合的感情状况返回JSON格式的结果。

#         自我肯定语: {row['自我肯定语']}
#         场景: {row['场景']}
#         子场景: {row['子场景']}
#         用户需求: {row['用户需求']}
#         心理作用机制与功能: {row['心理作用机制与功能']}
#         句子级别: {row['句子级别']}

#         1. 判断该自我肯定语及其场景描述是否适合以下感情状况的用户：
#         ["正在恋爱", "最近刚分手", "处在一段艰难的亲密关系中", "快乐地单身着", "单身但准备好了开始新的恋情", "有点辛苦的暗恋"]。
#         2. 选择一个或多个感情状况标签进行标注，**不要**选择其他标签。
#         3. 返回的标签列表应该准确反映与该自我肯定语的适配情况。
#         4. 如果不选择某些标签，则在"note"中给出理由。选择的标签不用给出。

#         请以如下格式返回结果：
#         {{
#             "key": "感情状况",
#             "value": ["正在恋爱", "单身但准备好了开始新的恋情"],
#             "note": "不放一些标签的理由"
#         }}
#         """
#     elif type == '最近的感觉':
#         message = f"""
#         请根据以下自我肯定语及其相关信息，判断它们是否适合推送给具有特定情感状态的用户，并根据用户的情感状态返回相应的JSON格式结果。

#         自我肯定语: {row['自我肯定语']}
#         场景: {row['场景']}
#         子场景: {row['子场景']}
#         用户需求: {row['用户需求']}
#         心理作用机制与功能: {row['心理作用机制与功能']}
#         句子级别: {row['句子级别']}

#         1. 判断该自我肯定语及其场景描述是否适合推送给以下情感状态的用户：
#         ["开心", "很好", "一般", "不好", "糟糕"]。
#         2. 请根据句子的情感倾向选择一个或多个标签：  
#         - 【开心】和【很好】可以同时出现，  
#         - 【不好】和【糟糕】也可以同时出现，  
#         - 不要选择额外的标签。
#         3. 1级句子倾向于【不好】和【糟糕】，2级句子倾向于【开心】和【很好】，但不是绝对的，具体分析。
#         4. "什么让你有这种感觉"是对"最近的感觉"的解释，请从以下选项中选择与感受相关的因素：
#         ["家庭", "朋友", "工作", "健康", "感情", "学业", "自己"]。
#         5. 如果不选择某些标签，则在"note"中给出理由。选择的标签不用给出。

#         请以如下格式返回结果：
#         {{
#             "最近的感觉": ["不好", "糟糕"],
#             "什么让你有这种感觉": ["家庭", "健康"],
#             "note": "不放一些标签的理由"
#         }}
#         """
#     return message

# def get_annotation(row, type='合集'):
#     role = f"noter-0205-{type}"
#     role_prompt = get_role_prompt(role=role)
    
#     # 根据不同类型生成提示词并返回相应的标注结果
#     content = get_content_by_type(row, type)
#     messages = [
#         {"role": "system", "content": role_prompt},
#         {"role": "user", "content": content}
#     ]
    
#     try:
#         completion = client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=messages,
#             temperature=1,
#             response_format={"type": "json_object"},  # 确保返回 JSON 格式
#             n=1  # 请求返回1个结果
#         )
#         response = completion.choices[0].message.content.strip()
#         response_dict = json.loads(response)
        
#         result = {}
#         if type == "合集":
#             result['合集'] = response_dict.get('value', [])
#             result['合集理由'] = response_dict.get('note', [])
#         elif type == "感情状况":
#             result['感情状况'] = response_dict.get('value', [])
#             result['感情状况理由'] = response_dict.get('note', [])
#         elif type == "最近的感觉":
#             result['最近的感觉'] = response_dict.get("最近的感觉", [])
#             result['什么让你有这种感觉'] = response_dict.get('什么让你有这种感觉', [])
#             result['最近的感觉理由'] = response_dict.get('note', [])
        
#         logger.info(f"成功获取{type}标注")
#         return result
#     except Exception as e:
#         logger.error(f"获取{type}标注时发生错误: {e}")
#         return {}

# def main():
#     input_path = "../data/0205标注协作表 - 上线句子清单0205.csv"
#     logger.info(f"开始加载数据集: {input_path}")
#     df = pd.read_csv(input_path)
#     df = df[['自我肯定语', '场景', '子场景', '用户需求', '心理作用机制与功能', '句子级别']]
#     # df = df.sample(1)
#     # 结果存储
#     results = []

#     # 使用tqdm包裹迭代器显示进度条
#     for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="处理数据"):
#         row_results = {}

#         # 将自我肯定语添加为字典项
#         row_results.update({'自我肯定语': row['自我肯定语']})
        
#         for type in ['合集', '感情状况', '最近的感觉']:
#             annotation_result = get_annotation(row, type)
#             row_results.update(annotation_result)
        
#         # 更新其它列
#         row_results.update({'场景': row['场景']})
#         row_results.update({'子场景': row['子场景']})
#         row_results.update({'用户需求': row['用户需求']})
#         row_results.update({'心理作用机制与功能': row['心理作用机制与功能']})
#         row_results.update({'句子级别': row['句子级别']})
        
#         results.append(row_results)

#     # 将结果保存到一个新的DataFrame并导出
#     result_df = pd.DataFrame(results)
#     print(result_df.columns)
#     result_df = result_df[["自我肯定语","合集","感情状况","最近的感觉","什么让你有这种感觉","合集理由","感情状况理由","最近的感觉理由","场景","子场景","用户需求","心理作用机制与功能","句子级别"]]
#     result_df.to_csv(input_path.replace('.csv', '_result.csv'), index=False)
#     logger.info(f"标注完成，结果已导出为{input_path.replace('.csv', '_result.csv')}")

# if __name__ == "__main__":
#     main()