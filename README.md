# Caritas 自我肯定语内容生产
![](https://caritas.pro/affirm/assets/images/icons/nav-icon.png)
APP链接：https://caritas.pro/affirm/ 

1. 配置`.env`
```
KIMI_API_KEY = sk-Ar
ZHIPU_API_key = 65
Token = b
```
2. 下载数据集：[百度网盘链接]()
内部使用，密码暂不公开
放置到`./data/`


### CMD
```bash
pip install -r requirements.txt 
make
# python main.py
make run
make clean

```

### kimi or Deepseek
```python
from src.kimi_api import client,MODEL_NAME # kimi
# from src.deepseek_api import client,MODEL_NAME # Deepseek
```


---

