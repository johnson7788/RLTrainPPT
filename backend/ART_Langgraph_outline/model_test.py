#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/27 11:41
# @File  : model_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 基于 LangGraph + ART 的“按 topic 搜索并生成 Markdown 大纲”测试脚本

import os
import uuid
import asyncio
from textwrap import dedent
from typing import List, Optional
import dotenv
import art
from art.langgraph import init_chat_model, wrap_rollout
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from zai import ZhipuAiClient

dotenv.load_dotenv()

# ---------- 与训练保持一致 ----------
NAME = os.getenv("ART_NAME", "web-search-outline")
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT_NAME = os.getenv("ART_PROJECT", "web-search-outline-training")
USE_LOCAL_BACKEND = os.getenv("ART_BACKEND", "local").lower() == "local"
WebSearchClient = ZhipuAiClient(api_key=os.environ["ZHIPU_API_KEY"])

# ---------- 业务工具 ----------
class WebSearchResult(BaseModel):
    url: str
    title: str
    snippet: str

def search_web(keyword: str) -> List[WebSearchResult]:
    """
    真实的网络搜索函数
    """
    response = WebSearchClient.web_search.web_search(
        search_engine="search_std",
        search_query=keyword,
        count=15,
        search_recency_filter="noLimit",
        content_size="high"
    )
    return [
        WebSearchResult(
            url=item["url"],
            title=item["title"],
            snippet=item["content"]
        ) for item in response["search_result"]
    ]


async def run_agent_test(model: art.Model, topic: str):
    """
    基于训练好的同名模型，按 topic 进行搜索并生成大纲
    """
    system_prompt = dedent("""
    你是“大纲生成智能体”。流程：先使用 web_search_tool 对 topic 进行检索与聚合，再输出中文 Markdown 大纲。
    必须严格遵守以下格式与规则：
    - 使用 Markdown 标题层级：# 标题 → ## 一级部分 → ### 二级小节 → 列表要点
    - 一级部分数量：5个；每个一级部分下含3–4个二级小节
    - 每个二级小节列出3–5个要点；要点使用短句、动词开头、不超过18字、不要句号
    - 全文不写引言/结语/目录，不写解释性段落，不加任何额外说明
    - 术语统一、风格一致，必要时加入可量化指标或示例
    - 语言：简体中文
    - 仅输出大纲本身，不输出“参考文献”等额外文字
    生成后，调用 return_final_outline_tool(outline, source_urls) 提交结果，其中 source_urls 为你参考的高质量链接（3-8条）。
    """)

    final_outline: Optional[str] = None

    @tool
    def web_search_tool(query: str) -> List[dict]:
        """根据查询词进行网络搜索，返回结果列表。"""
        results = search_web(query)
        return [r.model_dump() for r in results]

    @tool
    def return_final_outline_tool(outline: str, source_urls: List[str]) -> dict:
        """提交最终大纲以及引用来源 URL 列表。"""
        nonlocal final_outline
        final_outline = outline
        # 返回值仅用于可观测性
        return {"outline": outline, "source_urls": source_urls}

    tools = [web_search_tool, return_final_outline_tool]

    # 用 ART 的 init_chat_model 获取可用的聊天模型（后端会加载最近训练好的 LoRA）
    chat_model = init_chat_model(model, temperature=0.3)
    agent = create_react_agent(chat_model, tools)

    res = await agent.ainvoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"topic：{topic}\n请严格按规则生成大纲，并在完成后调用 return_final_outline_tool 提交。")
            ]
        },
        config={
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": 12
        },
    )
    print("====== 推理返回（含工具轨迹） ======")
    print(res)
    print("====== 最终大纲(如已提交) ======")
    if final_outline:
        print(final_outline)
    print("[TEST] agent finished. See backend logs / tracing for details.")


async def main():
    # 连接与注册后端
    if USE_LOCAL_BACKEND:
        from art.local.backend import LocalBackend
        backend = LocalBackend()
    else:
        from art.skypilot.backend import SkyPilotBackend
        backend = await SkyPilotBackend.initialize_cluster(
            cluster_name=os.getenv("ART_SKYPILOT_CLUSTER", "art-cluster"),
            gpu=os.getenv("ART_GPU", "A100"),
        )

    model = art.TrainableModel(name=NAME, project=PROJECT_NAME, base_model=MODEL_NAME)
    await model.register(backend)

    topic = os.getenv("OUTLINE_TOPIC") or "AIGC 在医疗影像的应用趋势"
    # 用 wrap_rollout 包装，确保 ART 上下文正确设置
    wrapped_test_func = wrap_rollout(model, run_agent_test)
    await wrapped_test_func(model, topic)

if __name__ == "__main__":
    asyncio.run(main())
