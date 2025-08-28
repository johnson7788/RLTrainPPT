# 基础镜像，选择较小的 Alpine 或 Debian
FROM debian:bullseye-slim

# 设置工作目录
WORKDIR /app

# 复制 clash 及配置文件
COPY clash-linux-amd64-v1.10.0 /app/clash
COPY config.yaml /app/config.yaml
COPY Country.mmdb /app/Country.mmdb
COPY dashboard/ /app/dashboard/

# 给执行权限
RUN chmod +x /app/clash

# 暴露默认端口
# HTTP/SOCKS5 代理
EXPOSE 7890
# RESTful 控制 API
EXPOSE 9090

# 启动 Clash
CMD ["/app/clash", "-d", "/app"]
