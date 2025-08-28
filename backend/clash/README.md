# clash 容器(用于代理Gemini等模型)

# 启动
./clash-linux-amd64-v1.10.0 -d .

# 容器启动
docker build -t clash .

# 在Agent下的.env中配置
```
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
```