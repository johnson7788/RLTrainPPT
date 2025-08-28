#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/6/22
# @File  : LLM_cache.py
# @Author: johnson
# @Desc  : 大语言模型代理，缓存到本地文件,
# 1. 大模型回退策略，当尝试max_retries，当大于3次时，自动使用fallback_model的模型
"""
同样可以在.env中配置:HTTP_PROXY
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890

curl -X POST http://localhost:6688/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
     -d '{
         "model": "qwen-turbo-latest",
         "messages": [
             {"role": "user", "content": "你好"}
         ],
         "stream": true
     }'
# 测试o3-mini模型
curl -X POST http://localhost:6688/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer xxx" \
     -d '{
         "model": "o3-mini",
         "messages": [
             {"role": "user", "content": "你好"}
         ],
         "stream": true
     }'
"""

import os
import httpx
import hashlib
import asyncio
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse
from fastapi.responses import PlainTextResponse
from google import genai  # pip install google-genai
import dotenv
import time

dotenv.load_dotenv('.env_llm_cache')

async def generate_google_streaming_text(prompt: str, api_key, model="gemini-2.0-flash"):
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    for chunk in client.models.generate_content_stream(
            model=model, contents=prompt
    ):
        print(chunk.text, end='')
        yield chunk.text


CACHE_DIR = "llm_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class AppLogger:
    def __init__(self, log_file="llm.log"):
        self.log_file = log_file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("")

    def log(self, message: str):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
        print(message)


app = FastAPI(title="LLM API Logger")
logger = AppLogger("llm.log")

# 模型名称对应的访问的base url， 注意chat/completions结尾哦
provider2url = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "qwen-turbo-latest": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "qwq-plus-latest": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "gpt-4.1": "https://api.openai.com/v1/chat/completions",
    "gpt-4.1-nano-2025-04-14": "https://api.openai.com/v1/chat/completions",
    "gpt-4o-mini": "https://api.openai.com/v1/chat/completions",
    "gpt-4o": "https://api.openai.com/v1/chat/completions",
    "o3-mini": "https://api.openai.com/v1/chat/completions",
    "o4-mini": "https://api.openai.com/v1/chat/completions",
    "o3": "https://api.openai.com/v1/chat/completions",
    "deepseek-chat": "https://api.deepseek.com/v1/chat/completions",
    "doubao-seed-1-6-flash-250615": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "doubao-seed-1-6-thinking-250715": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "doubao-seed-1-6-250615": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "deepseek-r1-250528": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "deepseek-v3-250324": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "claude-sonnet-4": "https://api.anthropic.com/v1",
}


def check_cache_for_errors(delete_error_files=True):
    print(f"检查缓存文件中是否存在错误内容...， 日志记录到llm.log和{CACHE_DIR}中")
    for filename in os.listdir(CACHE_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(CACHE_DIR, filename)
            try:
                find_errors = False
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "error" in content.lower():
                        print(f"⚠️  警告：缓存文件 {filename} 中包含 'error'，这会影响LLM")
                        find_errors = True
                if find_errors and delete_error_files:
                    os.remove(file_path)
                    print(f"已删除错误文件：{file_path}")
            except Exception as e:
                print(f"读取缓存文件 {filename} 失败：{e}")


def get_provider_url_by_model(model: str):
    """
    :param model:
    :return:
    """
    assert model in provider2url, f"{model} not support，请手动添加模型对应的供应商url,{provider2url}"
    url = provider2url.get(model)
    return url


def compute_hash(data: str) -> str:
    """计算 SHA256 哈希作为缓存键"""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def get_cache_path(hash_key: str) -> str:
    """返回缓存文件路径"""
    return os.path.join(CACHE_DIR, f"{hash_key}.txt")


@app.post("/chat/completions")
async def proxy_request(request: Request):
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8")
    logger.log(f"模型请求：{body_str}")
    body = await request.json()
    provider_url = get_provider_url_by_model(body["model"])
    cache_key = compute_hash(body_str)
    cache_path = get_cache_path(cache_key)
    request_cache_path = os.path.join(CACHE_DIR, f"{cache_key}.request")

    if not os.path.exists(request_cache_path):
        logger.log(f"记录请求信息到文件中：{request_cache_path}")
        with open(request_cache_path, 'w', encoding='utf-8') as f:
            f.write(body_str)

    # 读取缓存
    if os.path.exists(cache_path):
        logger.log(f"命中本地缓存：{cache_path}")

        async def cached_stream():
            with open(cache_path, 'r', encoding='utf-8') as f:
                for line in f:
                    print(f"缓存{cache_path}对应的内容是:{line[:100]}")
                    await asyncio.sleep(0.2)  # 每行间隔 0.2 秒
                    yield line  # 每行已含 \n

        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    logger.log(f"未命中缓存，开始请求模型{provider_url}的模型{body['model']}, 请求信息是总长度: {len(body_str)}")

    assert provider_url, "请检查模型名称是否正确,提供的模型是否有对应的链接？"

    async def event_stream(has_switched: bool = False):
        lines = []
        max_retries = 5
        retry_delay = 1
        fallback_model = "qwen-turbo-latest"

        nonlocal provider_url, body  # 允许修改外层变量

        for attempt in range(max_retries):
            try:
                logger.log(f"尝试连接LLM服务器 (第 {attempt + 1} 次)，当前模型: {body['model']}")

                timeout = httpx.Timeout(600.0, connect=20.0)
                async with httpx.AsyncClient(
                        timeout=timeout,
                        verify=False,
                        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
                ) as client:

                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                        "User-Agent": "LLM-Cache-Proxy/1.0",
                        "Connection": "keep-alive"
                    }

                    if request.headers.get("Authorization"):
                        headers["Authorization"] = request.headers.get("Authorization")
                    if has_switched and fallback_model == "qwen-turbo-latest":
                        headers["Authorization"] = os.environ["ALI_API_KEY"]

                    async with client.stream(
                            "POST",
                            provider_url,
                            json=body,
                            headers=headers,
                    ) as response:
                        logger.log(f"收到响应状态码: {response.status_code}")

                        if response.status_code != 200:
                            error_text = await response.aread()
                            logger.log(f"请求失败，状态码: {response.status_code}, 错误: {error_text}")
                            if attempt < max_retries - 1:
                                logger.log(f"等待 {retry_delay} 秒后重试...")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2
                                continue
                            else:
                                logger.log(f"超过最大重试次数，将尝试切换备用模型: {fallback_model}")
                                break

                        async for line in response.aiter_lines():
                            # if line.strip():  # 不能去掉换行，否则会产生Bug
                            logger.log(f"收到数据: {line}")
                            lines.append(line + "\n")
                            yield line + "\n"
                        break

            except Exception as e:
                logger.log(f"请求异常 (第 {attempt + 1} 次): {e}")
                if attempt < max_retries - 1:
                    logger.log(f"等待 {retry_delay} 秒后重试...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.log(f"超过最大重试次数，将尝试切换备用模型: {fallback_model}")
                    break

        # 如果所有尝试都失败，并且尚未切换过备用模型
        if not lines and not has_switched:
            try:
                body["model"] = fallback_model
                provider_url = get_provider_url_by_model(fallback_model)
                logger.log(f"切换到备用模型: {fallback_model}，URL: {provider_url}")
                async for chunk in event_stream(has_switched=True):  # 传入状态，防止无限递归
                    yield chunk
                return
            except Exception as e:
                error_msg = f"data: 'error': '备用模型请求失败: {str(e)}'\n\n"
                yield error_msg
                return

        if lines:
            content_str = "".join(lines)
            if any(err in content_str.lower() for err in ["incorrect api key", "timeout", "error"]):
                logger.log(f"⚠️ 缓存中含有错误关键词，跳过写入: {content_str}")
                return
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                logger.log(f"✅ 缓存写入成功：{cache_path}")
            except Exception as e:
                logger.log(f"缓存写入失败: {e}")


    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def unsupported_path(request: Request, path_name: str):
    logger.log(f"不支持的路径访问: {request.method} {request.url.path}")
    return PlainTextResponse("错误：不支持的路径", status_code=404)


# 启动时检查缓存中是否包含 "error"
check_cache_for_errors()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6688)