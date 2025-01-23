# I-am
Caritas 自我肯定语内容生产

`.env`
```
KIMI_API_KEY = sk-Ar
ZHIPU_API_key = 65
Token = b
```

```bash
pip install -r requirements.txt 
python main.py
```

## Log

```json
[
    {
        'id': 3639, 
        'distance': 0.46352270245552063, 
        'entity': {
            'last_update': '2021-04-11', 
            'content': '', 
            'read_count': 247, 
            'zhihu_link': 'https://www.zhihu.com/pin/1364700155837534208', 
            'author': '#Anonymity', 
            'question': '不知有多少鼓励的话，其实在将来会逼人自…', 
            'links': '', 
            'tags': '["想法集"]', 
            'id': 3639, 
            'favorites_count': 20, 
            'title': ''
        }, 
        'model': 'article'
    }, 
    {
        'id': 1644, 
        'distance': 0.3170081675052643, 
        'entity': {
            'last_update': '2021-09-05', 
            'content': '', 
            'read_count': 1230, 
            'zhihu_link': 'https://www.zhihu.com/question/484883057/answer/2104662616', 
            'author': '#Anonymity', 
            'question': '我们有没有资格安慰别人？如果有，该如何安慰？', 
            'links': '["同情的资格","以弱悯强","误解","赞美","批评与赞美"]', 
            'tags': '["1-家族/1B-处世/2-外务/2d-文责意识","1-家族/1C-安全/2-救助ta人","3-信仰/3A-Caritas/1-爱","1-家族/1E-两性/1-亲密关系/1a-交往准则/社交红线","1-家族/1A-内外/1-内在建设/1f-概念定义","处世"]', 
            'id': 1644, 
            'favorites_count': 84, 
            'title': '安慰'
        }, 
        'model': 'article'
    }
]

```

### 2025-01-23 
1. ✅ Macbook VScode SSH服务器 // 使用Github + 本地VScode/Codeserver
2. ✅ Response is not valid JSON // 使用多轮对话+短Prompt
3. ✅ CoT 光思不改，反思不正确//使用多轮对话
4. ✅ 并发之后不知道进度条
5. ✅ 逻辑拆解、理解材料，结合材料使用其中的事实描述和价值观逻辑来生成孤立语，去掉因为，
6. ✅ 作者风格：遍历一遍各种作者 余华，季羡林，老舍，鲁迅，



### 2025-01-22
1. ✅DEBUG=False时出现死锁
2. ✅症状 -> 需求 
3. ✅只删不增 不一定要求5条
4. 反思出来有问题但是不改：写一版思考日志的详细模板，重复每个问题，下指令要求修改
5. 季羡林
6. 翻译腔 改不掉

⚠️
1. 太多Response is not valid JSON了
    - ![alt text](./Log/image/{21EE2D5A-4CD0-4D35-8A09-D919491056EE}.png)
2. Macbook VScode SSH服务器
3. CoT 光思不改，反思不正确//不反思了？
4. 并发之后不知道进度条
5. 检索到的金句不是文章原文
原文不适合摘抄 逻辑、视角、
逻辑拆解、理解材料，结合材料使用其中的事实描述和价值观逻辑来生成孤立语，去掉因为，
6. 作者风格：遍历一遍各种作者 余华，季羡林，老舍，鲁迅，
7. 多轮对话，拆分思考

解读：原文转换出来的对句子的解读，案例，论证为什么xxx。第一人称该写的宣言？

### 2025-01-21 
1. ✅高级生产员
2. ✅引入文章
3. ✅文风 kimi qwen 柔和 柔美 
4. ✅5001：鼓励语生成器 批评过滤器
5. 重启，❌VScode SSH

1. 太多Response is not valid JSON了
2. Macbook VScode SSH服务器
3. CoT 光思不改，反思不正确//不反思了？
4. 并发之后不知道进度条
5. DEBUG=False时出现死锁

问题：
1. 检索到的金句：不是原文
2. 余华风格：会增加负面描述：对齐、识别出来、删掉
3. 反思出来有问题但是不改：写一版思考日志的详细模板，重复每个问题，下指令要求修改
4. 强制5条会无中生有：只删不增 不一定要求5条

ToDo
1. 反思日志导出为结果
2. 
症状 -> 需求 
负面 -> 正面 

句子级别的差异：(1)被安慰/(2)被鼓励改变和行动 -> 有效性

试一下 只用【子场景用户1级需求合并	子场景用户2级需求合并】生成，取消症状

5个1级需求  5个2级需求  这是1/2级需求
安慰效果    鼓励效果


余华没救了，不用余华了。删掉