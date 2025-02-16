from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import importlib.util
import os
import json

app = Flask(__name__)
CORS(app) 

PROMPT_FILE = os.path.join(os.path.dirname(__file__), '../prompt.py')
JSON_FILE = os.path.join(os.path.dirname(__file__), '../data/prompts.json')

def load_prompts():
    spec = importlib.util.spec_from_file_location("prompt", PROMPT_FILE)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    with open(JSON_FILE, 'r', encoding='utf-8') as json_file:
        json_prompts = json.load(json_file)
    return {**prompt_module.prompts, **json_prompts}

def save_prompts(role, content):
    # 自动创建 JSON 文件
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    
    with open(JSON_FILE, 'r+', encoding='utf-8') as json_file:
        data = json.load(json_file)
        data[role] = content
        json_file.seek(0)
        json.dump(data, json_file, ensure_ascii=False, indent=4)
        json_file.truncate()


@app.route('/')
def index():
    debug_status = app.debug
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prompt API Status</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            h1 { color: green; }
            .status { font-size: 18px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>🚀 Prompt API is Running!</h1>
        <div class="status">Debug Mode: <strong>{{ debug_status }}</strong></div>
    </body>
    </html>
    ''', debug_status=debug_status)

@app.route('/prompts', methods=['GET'])
def get_prompts():
    prompts = load_prompts()
    return jsonify(prompts)

@app.route('/prompts/<role>', methods=['GET'])
def get_prompt(role):
    prompts = load_prompts()
    return jsonify({role: prompts.get(role, "Prompt not found")})

@app.route('/prompts', methods=['POST'])
def update_prompt():
    data = request.get_json()
    if 'role' not in data or 'content' not in data:
        return jsonify({"error": "Invalid data format. 'role' and 'content' are required."}), 400
    save_prompts(data['role'], data['content'])
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)