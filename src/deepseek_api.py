import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

# (0) Official
# Deepseek_KEY = os.getenv('Deepseek_KEY')
# Deepseek_URL = "https://api.deepseek.com"
# client = OpenAI(api_key=Deepseek_KEY, base_url=Deepseek_URL)
# MODEL_NAME = "deepseek-chat"

# (1)
# LUCHENTECH_KEY = os.getenv('LUCHENTECH_KEY')
# LUCHENTECH_URL = 'https://cloud.luchentech.com/api/maas/chat/completions'

# (2) Tencent
tencent_KEY = os.getenv('tencent_KEY')
tencent_KEY_wzj  = os.getenv('tencent_KEY_wzj')
tencent_URL = "https://api.lkeap.cloud.tencent.com/v1"
client = OpenAI(api_key=tencent_KEY_wzj, base_url=tencent_URL)
MODEL_NAME = "deepseek-r1"


# Utils
def send_messages(messages, model=MODEL_NAME, tools=None):
    """
    Sends a list of messages to the chat completion API and returns the response.

    Args:
        messages (list): A list of message dictionaries to be sent to the API.
        model (str, optional): The model name to be used for the chat completion. Defaults to MODEL_NAME.
        tools (optional): Additional tools or parameters to be used by the API. Defaults to None.

    Returns:
        dict: The response message from the API.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools
    )
    return response.choices[0].message

# Usage example
if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ]
    response = send_messages(messages)
    print(response)
    

