import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
# LUCHENTECH_KEY = os.getenv('LUCHENTECH_KEY')
# LUCHENTECH_URL = 'https://cloud.luchentech.com/api/maas/chat/completions'

Deepseek_KEY = os.getenv('Deepseek_KEY')
Deepseek_URL = "https://api.deepseek.com"
client = OpenAI(api_key=Deepseek_KEY, base_url=Deepseek_URL)
MODEL_NAME = "deepseek-chat"