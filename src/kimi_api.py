import os
from dotenv import load_dotenv
from openai import OpenAI
# 从 .env 文件中加载环境变量
load_dotenv()

KIMI_API_KEY = os.getenv("KIMI_API_KEY")
BASE_URL = "https://api.moonshot.cn/v1"
client = OpenAI(api_key=KIMI_API_KEY, base_url=BASE_URL)