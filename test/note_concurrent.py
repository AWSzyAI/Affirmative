import pandas as pd
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append('../src')
from kimi_api import client

# 读取待处理数据
df = pd.read_csv('../data/协作表 - V1 - 待审查.csv')
df = df[['自我肯定语-V1', '字数']]

role_prompt = """
你是一个情感分析专家，负责对句子进行四个维度的标注。请务必全面考虑句子的语义，确保每个可能的选项都被评估，避免漏选。

1. 合集
选项：句子特征
   - 情感疗愈: 语义涉及情感修复、亲密关系或情感联结，即使未直接提及“情感”或“亲密关系”。
   - 自信: 语义涉及自我肯定、力量感或自我认知的提升，即使未直接提及“我能”或类似词语。
   - 平和心境: 语义涉及情绪调节、内心安宁或对现状的满足感，即使未直接提及“平和”、“安宁”或“感恩”。
   - 治愈之旅: 语义涉及自我接纳、自我宽慰或内在疗愈，即使未直接提及“安慰”或“接纳”。
   - 人际交往: 语义涉及与他人互动、沟通或关系建立，即使未直接提及“他人”或“沟通”，“联系”。
   - 自我关怀: 语义涉及自我爱护、身心健康或生活平衡，即使未直接提及“爱自己”、“健康”或“休息”。
   - 个人成长: 语义涉及自我提升、事业发展或学习创造，即使未直接提及“成长”、“工作”或“学习”。
   - 成为英雄: 语义涉及勇敢行动、宽恕他人、追求理想，付出、牺牲。即使未直接提及“勇气”、“原谅”或“理想”。

2. 感情状况
   这个句子适合推送给哪些感情状态的用户？请务必全面评估句子的语义，确保每个可能的选项都被考虑。
   选项：
   - 正在恋爱: 语义涉及甜蜜、稳定、幸福的亲密关系，或对伴侣的积极情感表达。
   - 处在一段艰难的亲密关系中: 语义涉及关系中的矛盾、困惑、挣扎，或对关系改善的渴望。可能会自卑、自我怀疑、因此也需要提升自信相关的句子
     - 排除：个人成长、成为英雄合集的句子。
   - 快乐地单身着: 语义涉及享受独处、自我成长、独立自主，或对单身状态的积极态度。
   - 单身但准备好了开始新的恋情: 语义涉及对爱情的期待、开放心态，或为新的关系做准备的积极情绪。
   - 最近刚分手: 语义涉及情感失落、疗愈、反思，或对过去关系的释怀。
     - 排除：个人成长、成为英雄合集的句子。
   - 有点辛苦的暗恋: 避免离开、分离，最好提及人际交往合集中涉及勇气的句子，疗愈、鼓励、尤其是人际关系上的主动
     - 排除：个人成长、成为英雄合集的句子。

   **注意**：如果一个句子同时适合多个感情状态，请务必全部选择。例如：
   - 句子：“爱自己是一切美好的开始。”  
     - 适合：**快乐地单身着**（享受独处）、**最近刚分手**（疗愈）、**单身但准备好了开始新的恋情**（积极心态）。

3. 最近的感觉
   选项：句子特征
   - 开心: 语义涉及自信、力量感、成就感（自信合集、个人成长合集）。
     - 排除：治愈之旅、平和心境合集的句子。
   - 很好: 语义涉及积极情绪、满足感或幸福感。
     - 排除：治愈之旅合集的句子。
   - 一般: 语义涉及中性情绪或日常状态，包括自信。
     - 排除：成为英雄合集的句子。
   - 不好: 语义涉及需要安慰、支持或情绪调节（治愈之旅合集、平和心境合集）,不排除自信。
     - 排除：个人成长、成为英雄合集的句子。
   - 糟糕: 语义涉及强烈的情感低落、痛苦或无助感（治愈之旅合集、平和心境合集）。
     - 排除：个人成长、成为英雄合集的句子。

   **注意**：如果一个句子同时适合多个感觉，请务必全部选择。例如：
   - 句子：“最近总是觉得很累，需要好好休息一下。”  
     - 适合：**不好**（需要安慰）、**糟糕**（强烈情感低落）。
    

4. 什么让你有这种感觉（多选）
   选项：句子特征
   - 家庭: 包含“家”和人际关系合集的句子。
   - 朋友: 人际关系合集的句子。
   - 工作: 个人成长合集中提及事业、工作的句子。
   - 健康: 自我关怀合集的句子。
   - 感情: 提及“爱”、“情感”、“亲密关系”
   - 学业: 个人成长合集中提及学习的句子。
     - 排除：工作、事业相关的句子。
   - 自己: 自信、自我关怀合集的句子

   **注意**：如果一个句子同时涉及多个原因，请务必全部选择。例如：
    - 句子：“最近工作压力大，但朋友的支持让我感到温暖。”  
        - 适合：**工作**（压力）、**朋友**（支持）。
    - 并不只有在感觉不好/糟糕的时候才分析是学业/工作产生的，当用户觉得开心、很好的时候，也可能是学业/工作上有所成就造成的。
"""

# 定义标注函数（带重试机制）
def get_annotations(sentence, i=None, j=None, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            if i is None and j is None:
                message = f"""
                    句子内容: {sentence}
                    
                    从"情感疗愈","自信","平和心境","治愈之旅","人际交往","自我关怀","个人成长","成为英雄" 选一个，不要捏造其他的标签。

                    请根据提供的句子特征进行标注，用以匹配其适用的场景，并返回JSON格式的结果。以下是返回结果的JSON示例：
                    {{
                        "合集": "情感疗愈",
                    }}
                """
            else:
                message = f"""
                句子 "{sentence}" 是否适合推荐给情感状况为 {i}, 心情为 {j} 的用户？
                return 
                {{
                    "感情状况": "{i}",
                    "最近的感觉": "{j}",
                    "什么让你有这种感觉": ["家庭","朋友","工作","健康","感情","学业","自己"], #分析{sentence}得到
                    "suitable": True  # 或 False，根据需要调整
                    "exlapin": "False 的理由"
                }}
                """
            
            messages = [
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": message}
            ]
            
            completion = client.chat.completions.create(
                model="moonshot-v1-auto",
                messages=messages,
                temperature=1,
                response_format={"type": "json_object"},  # 确保返回 JSON 格式
                n=1  # 请求返回1个结果
            )
            response = completion.choices[0].message.content.strip()
            response_dict = json.loads(response)
            return response_dict
        except Exception as e:
            if "rate_limit_reached_error" in str(e):
                retries += 1
                print(f"Rate limit reached. Retrying ({retries}/{max_retries})...")
                time.sleep(1)  # 减少重试间隔到 1 秒
            else:
                print(f"Error: {e}")
                return {"error": str(e)}
    print("Max retries reached")
    return {"error": "Max retries reached"}

# 并发处理函数
def process_sentence(sentence):
    results = {
        "句子": sentence,
        "合集": "",
        "感情状况": [],
        "最近的感觉": [],
        "什么让你有这种感觉": [],
    }
    
    # 获取合集标注
    annotations = get_annotations(sentence)
    results["合集"] = annotations.get("合集", "")
    
    # 并发获取感情状况和最近的感觉标注
    with ThreadPoolExecutor(max_workers=4) as executor:  # 增加并发数到 4
        futures = []
        for i in ["正在恋爱", "处在一段艰难的亲密关系中", "快乐地单身着", "单身但准备好了开始新的恋情", "最近刚分手", "有点辛苦的暗恋"]:
            for j in ["开心/很好", "一般", "不好/糟糕"]:
                futures.append(executor.submit(get_annotations, sentence, i, j))
                time.sleep(0.5)  # 减少等待时间到 0.5 秒
        
        for future in as_completed(futures):
            annotations = future.result()
            if annotations.get("suitable", False):
                i = annotations.get("感情状况", "")
                j = annotations.get("最近的感觉", "")
                if i not in results["感情状况"]:
                    results["感情状况"].append(i)
                if j not in results["最近的感觉"]:
                    results["最近的感觉"].append(j)
                results["什么让你有这种感觉"].extend(annotations.get("什么让你有这种感觉", []))
    
    return results

# 存储标注结果
batch_size = 100  # 每批次处理 100 条数据
results = []

for start in range(0, len(df), batch_size):
    end = start + batch_size
    batch_df = df[start:end]
    
    with ThreadPoolExecutor(max_workers=4) as executor:  # 增加并发数到 4
        futures = [executor.submit(process_sentence, row['自我肯定语-V1']) for _, row in batch_df.iterrows()]
        for future in tqdm(as_completed(futures), total=len(batch_df), desc=f"处理批次 {start//batch_size + 1}"):
            results.append(future.result())
    
    if end < len(df):  # 如果不是最后一批，等待一段时间
        print("等待 5 秒以缓解速率限制...")
        # 此时保存一下数据
        batch_results_df = pd.DataFrame(results)
        batch_output_file = f'../data/con标注结果_batch_{start//batch_size + 1}.csv'
        batch_results_df.to_csv(batch_output_file, index=False)
        print(f"当前批次结果已保存为 {batch_output_file}")
        time.sleep(5)  # 减少等待时间到 1 秒

# 导出结果
results_df = pd.DataFrame(results)
output_file = f'../data/con标注结果_batch_1.csv'
results_df.to_csv(output_file, index=False)
print(f"标注完成，结果已导出为 {output_file}")