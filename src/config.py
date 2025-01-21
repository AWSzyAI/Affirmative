import os
from dotenv import load_dotenv

# 从 .env 文件中加载环境变量
load_dotenv()

KIMI_API_KEY = os.getenv("KIMI_API_KEY")
BASE_URL = "https://api.moonshot.cn/v1"
