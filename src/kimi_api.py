import os
from dotenv import load_dotenv
from openai import OpenAI
# 从 .env 文件中加载环境变量
load_dotenv()

KIMI_API_KEY = os.getenv("KIMI_API_KEY")
BASE_URL = "https://api.moonshot.cn/v1"
# BASE_URL = "http://127.0.0.1:11434/v1/"
client = OpenAI(api_key=KIMI_API_KEY, base_url=BASE_URL)
MODEL_NAME = "moonshot-v1-auto"
# MODEL_NAME = "deepseek-r1:14b"

