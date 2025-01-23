import json
from src.prompt import get_role_prompt

def structured_articles_from_articles_data(article_data, client):
    """
    将输入的 article_data 转换为结构化的 JSON 格式。

    :param article_data: 输入的原始文章数据，格式为 article['entity']['content']
    :param client: API 客户端对象，用于调用模型
    :return: 结构化的文章数据，格式为 JSON
    """
    structured_articles = []

    # 系统提示词，用于指导模型生成结构化内容
    role_prompt = get_role_prompt("article-structurer")

    for article in article_data:
        content = article['entity']['content']

        # 调用 API 进行结构化转换
        completion = client.chat.completions.create(
            model="moonshot-v1-auto",
            messages=[
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": content}
            ],
            temperature=1,
            response_format={"type": "json_object"},  # 确保返回 JSON 格式
            n=1  # 请求返回1个结果
        )

        # 解析 API 返回的内容
        response = completion.choices[0].message.content.strip()
        try:
            structured_article = json.loads(response)  # 将字符串解析为 JSON 对象
            structured_articles.append(structured_article)
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON response for article: {content}")

    return structured_articles

    
