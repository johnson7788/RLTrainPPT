# clash 容器(用于代理Gemini等模型)

# 启动
./clash-linux-amd64-v1.10.0 -d .

# config.yml是VPN的配置文件，请拷贝你的Clash的电脑上的相同的配置文件到这里，进行替换
请手动替换config.yml文件

# 容器启动
docker build -t clash .

# 在Agent下的.env中配置
```
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
```