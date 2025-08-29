#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/27 11:30
# @File  : train.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 训练“按 topic 搜索并生成 Markdown 大纲”的 LangGraph ReAct Agent（ART GRPO）

import os
import uuid
import time
import asyncio
from statistics import mean
from textwrap import dedent
from typing import List, Optional
import re
import dotenv
import art
from art.langgraph import init_chat_model, wrap_rollout
from art.utils import iterate_dataset
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt
from litellm import acompletion
from zai import ZhipuAiClient

# ---------------- wandb ----------------
import wandb

dotenv.load_dotenv()

# ---------- 配置 ----------
NAME = os.getenv("ART_NAME", "web-search-outline")
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT_NAME = os.getenv("ART_PROJECT", "web-search-outline-training")
USE_LOCAL_BACKEND = os.getenv("ART_BACKEND", "local").lower() == "local"

print(f"{NAME} - {MODEL_NAME} - {PROJECT_NAME}")

# RULER 评估模型（可选；需相应 API Key）
RULER_MODEL = os.getenv("RULER_MODEL", "openai/o4-mini")

# wandb
WANDB_PROJECT = os.getenv("WANDB_PROJECT", PROJECT_NAME)
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", f"{NAME}-{time.strftime('%Y%m%d-%H%M%S')}")

WebSearchClient = ZhipuAiClient(api_key=os.environ["ZHIPU_API_KEY"])

# ---------- 数据结构 ----------
class WebSearchResult(BaseModel):
    url: str
    title: str
    snippet: str

class FinalOutline(BaseModel):
    outline: str
    source_urls: List[str]

class Scenario(BaseModel):
    id: str
    topic: str
    # 参考答案可留空；训练以相对评分/结构校验为主
    reference_outline: Optional[str] = None

class WebSearchScenario(BaseModel):
    step: int
    scenario: Scenario

class ProjectTrajectory(art.Trajectory):
    final_outline: Optional[FinalOutline] = None

# ---------- 搜索 ----------
async def search_web(keyword: str) -> List[WebSearchResult]:
    response = WebSearchClient.web_search.web_search(
        search_engine="search_std",
        search_query=keyword,
        count=15,
        search_recency_filter="noLimit",
        content_size="high"
    )
    if not response.search_result:
        return []

    return [
        WebSearchResult(
            url=sr.link,
            title=sr.title,
            snippet=sr.content
        )
        for sr in response.search_result
    ]

# ---------- 简单结构打分（可选：用于日志） ----------
def _structure_score_cn(md: str) -> float:
    """
    纯规则校验，返回 [0,1] 分：
    - # 1个
    - ## 恰好5个
    - 每个 ## 下有 3-4 个 ###
    - 每个 ### 下有 3-5 个 “- ”要点行
    - 不含“参考”“结语”“目录”等额外段落
    """
    try:
        # 仅一行一级标题
        h1 = re.findall(r"(?m)^# [^\n]+$", md)
        if len(h1) != 1:
            return 0.0
        # 二级标题
        h2_positions = [(m.start(), m.group()) for m in re.finditer(r"(?m)^## [^\n]+$", md)]
        if len(h2_positions) != 5:
            return 0.2
        h2_positions.append((len(md), ""))  # 便于切片
        per_h2_pass = 0
        total_h3 = 0
        total_bullets = 0

        for i in range(5):
            start = h2_positions[i][0]
            end = h2_positions[i+1][0]
            block = md[start:end]

            h3s = list(re.finditer(r"(?m)^### [^\n]+$", block))
            if len(h3s) < 3 or len(h3s) > 4:
                continue
            # 统计每个 h3 下的要点数量
            ok_h3 = 0
            for j, h3 in enumerate(h3s):
                b_start = h3.end()
                b_end = h3s[j+1].start() if j+1 < len(h3s) else len(block)
                sub = block[b_start:b_end]
                bullets = re.findall(r"(?m)^- [^\n]+$", sub)
                # 约束：3-5条，短句，动词开头，不以句号结尾
                if 3 <= len(bullets) <= 5 and all(
                    len(b) <= len("- ") + 18 and not b.endswith(("。", ".", "．"))
                    for b in [x[2:] for x in bullets]  # 去掉 "- "
                ):
                    ok_h3 += 1
                    total_bullets += len(bullets)
            if ok_h3 == len(h3s):
                per_h2_pass += 1
            total_h3 += len(h3s)

        score = 0.2  # 通过基本检查
        score += 0.3 * (per_h2_pass / 5.0)
        score += 0.3 * min(1.0, total_h3 / 18.0)  # 期望 5*(3~4)=15~20
        score += 0.2 * min(1.0, total_bullets / 60.0)  # 粗略目标 20*3=60 起步
        # 负面关键词惩罚
        if re.search(r"(参考|结语|总结|目录|说明|免责声明)", md):
            score *= 0.7
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0

class CorrectnessJudgeResponse(BaseModel):
    reasoning: str = Field(description="why")
    accept: bool = Field(description="是否满足结构与中文规范")

@retry(stop=stop_after_attempt(3))
async def judge_correctness(s: Scenario, outline: str) -> CorrectnessJudgeResponse:
    """
    使用一个小模型进行结构性与格式性判断（可选，仅做日志展示）。
    """
    system_prompt = "作为评审，请判断以下 Markdown 是否满足给定结构与中文书写规范，仅输出 JSON。"
    user = dedent(f"""
    主题: {s.topic}
    大纲:
    {outline}

    判定要点：
    1) 标题层级：# 1个；## 恰好5个；每个##下有3–4个###；每个###下有3–5个以“- ”开头的要点；
    2) 要点短句、动词开头、不超过18字；不以句号结尾；
    3) 仅输出大纲，无引言/结语/目录/说明等；
    4) 语言为简体中文；术语统一；允许包含可量化指标或示例。
    如果基本满足则 accept=true，否则为 false，并说明原因。
    """).strip()
    try:
        resp = await acompletion(
            model="openai/gpt-4o-mini",
            base_url="http://127.0.0.1:6688",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user}],
            response_format=CorrectnessJudgeResponse,
        )
        return CorrectnessJudgeResponse.model_validate_json(
            resp.choices[0].message.content or "{}"
        )
    except Exception:
        # 回退到规则打分
        score = _structure_score_cn(outline)
        return CorrectnessJudgeResponse(reasoning=f"rule_score={score:.2f}", accept=score >= 0.7)

# ---------- rollout：LangGraph + Tools ----------
async def rollout(model: art.Model, web_search_scenario: WebSearchScenario) -> ProjectTrajectory:
    scenario = web_search_scenario.scenario
    MAX_TURNS = 10

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={"scenario_id": scenario.id, "step": web_search_scenario.step},
    )

    system_prompt = dedent("""
    你是“大纲生成智能体”。目标：围绕用户给定的 topic，先使用 web_search_tool 进行信息检索与综合，再生成“中文 Markdown 大纲”。
    必须严格遵循：
    - 使用 Markdown 层级：# 标题 → ## 一级部分 → ### 二级小节 → 列表要点
    - 一级部分数量：5个；每个一级部分下含3–4个二级小节
    - 每个二级小节列出3–5个要点；要点使用短句、动词开头、不超过18字、不要句号
    - 全文不写引言/结语/目录，不写解释性段落，不加任何额外说明
    - 术语统一、风格一致，必要时加入可量化指标或示例
    - 输出语言：简体中文
    生成后必须调用 return_final_outline_tool(outline, source_urls) 提交结果，source_urls 提供3–8条高质量来源链接。
    """)

    final_outline: Optional[FinalOutline] = None

    @tool
    async def web_search_tool(query: str) -> List[dict]:
        """进行网络搜索并返回结果列表。"""
        print(f"[tool:web_search] scenario_id={scenario.id} step={web_search_scenario.step} query={query}")
        results = await search_web(query)
        print(f"[tool:web_search] results={results}")
        return [r.model_dump() for r in results]

    @tool
    def return_final_outline_tool(outline: str, source_urls: List[str]) -> dict:
        """提交最终大纲与来源链接。"""
        nonlocal final_outline
        final_outline = FinalOutline(outline=outline, source_urls=source_urls)
        return final_outline.model_dump()

    tools = [web_search_tool, return_final_outline_tool]

    # 用 ART 的 init_chat_model 注入可训练聊天模型
    chat_model = init_chat_model(MODEL_NAME, temperature=0.4)
    agent = create_react_agent(chat_model, tools)
    print(f"[rollout] START scenario_id={scenario.id} step={web_search_scenario.step} topic={scenario.topic}")

    await agent.ainvoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"topic：{scenario.topic}\n请严格按规则生成大纲，并在完成后调用 return_final_outline_tool 提交。")
            ]
        },
        config={"configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": MAX_TURNS},
    )

    if final_outline:
        traj.final_outline = final_outline
        print("[rollout] OUTLINE_PREVIEW ↓↓↓")
        print(f"final_outline: {final_outline}")
        print("[rollout] SOURCES:", ", ".join(final_outline.source_urls))
        try:
            judge = await judge_correctness(scenario, final_outline.outline)
            traj.metrics["pass_structure"] = float(judge.accept)
            # 将规则分也记录到 metrics，便于 wandb 可视化
            traj.metrics["rule_score"] = _structure_score_cn(final_outline.outline)
            print(f"[rollout] METRICS pass_structure={traj.metrics['pass_structure']} rule_score={traj.metrics['rule_score']:.2f}")
        except Exception:
            pass

    return traj

# ---------------- wandb: 日志封装 ----------------
def _log_batch_to_wandb(*, batch, finished_groups, use_ruler: bool):
    trajectories = []
    for g in finished_groups:
        if hasattr(g, "trajectories"):
            trajectories.extend(getattr(g, "trajectories"))
        else:
            try:
                trajectories.extend(list(g))
            except Exception:
                pass

    num_traj = len(trajectories)
    num_with_final = sum(1 for t in trajectories if getattr(t, "final_outline", None))
    pass_vals = []
    rule_scores = []
    for t in trajectories:
        m = getattr(t, "metrics", None)
        if isinstance(m, dict):
            if "pass_structure" in m:
                try:
                    pass_vals.append(float(m["pass_structure"]))
                except Exception:
                    pass
            if "rule_score" in m:
                try:
                    rule_scores.append(float(m["rule_score"]))
                except Exception:
                    pass

    pass_rate = mean(pass_vals) if pass_vals else 0.0
    avg_rule = mean(rule_scores) if rule_scores else 0.0
    coverage = (num_with_final / num_traj) if num_traj else 0.0

    try:
        table = wandb.Table(columns=["scenario_id", "topic", "outline_preview", "sources"])
        for t in trajectories[:40]:
            meta = getattr(t, "metadata", {}) or {}
            s_id = meta.get("scenario_id", "")
            topic = ""
            try:
                for s in batch.items:
                    if s.id == s_id:
                        topic = s.topic
                        break
            except Exception:
                pass
            fo = getattr(t, "final_outline", None)
            outline_preview = (getattr(fo, "outline", "") or "")[:500]
            srcs = ", ".join(getattr(fo, "source_urls", []) if fo else [])
            table.add_data(s_id, topic, outline_preview, srcs)
    except Exception:
        table = None

    log_dict = {
        "train/step": batch.step,
        "train/epoch": batch.epoch,
        "ruler/enabled": int(bool(use_ruler)),
        "data/num_trajectories": num_traj,
        "data/final_outline_coverage": coverage,
        "eval/pass_structure_rate": pass_rate,
        "eval/rule_score_avg": avg_rule,
    }
    if table is not None:
        log_dict["samples/rollouts"] = table

    wandb.log(log_dict, step=batch.step)

# ---------- 训练主程序 ----------
async def main():
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY if WANDB_ENTITY else None,
        name=WANDB_RUN_NAME,
        config={
            "art_project": PROJECT_NAME,
            "art_name": NAME,
            "base_model": MODEL_NAME,
            "backend": "local" if USE_LOCAL_BACKEND else "skypilot",
            "ruler_model": RULER_MODEL,
        },
        settings=wandb.Settings(start_method="thread"),
    )
    wandb.define_metric("*", step_metric="train/step")

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

    # 训练集：仅提供 topic，reference_outline 可选留空（训练采用相对比较/结构评估）
    training_scenarios = [
        Scenario(id="1", topic="AIGC 在医疗影像的应用与合规"),
        Scenario(id="2", topic="边缘计算在智能制造的实施路径"),
        Scenario(id="3", topic="量子计算基础与典型算法概览"),
        Scenario(id="4", topic="RAG 在企业知识库的效果评估"),
    ]

    # 训练参数
    training_config = {
        "groups_per_step": 2,
        "num_epochs": 2,
        "rollouts_per_group": 4,
        "learning_rate": 1e-5,
        "max_steps": 5,
    }

    # wandb 数据概览
    try:
        scen_table = wandb.Table(columns=["id", "topic"])
        for s in training_scenarios:
            scen_table.add_data(s.id, s.topic)
        wandb.log({"data/training_scenarios": scen_table}, step=0)
    except Exception:
        pass

    it = iterate_dataset(
        training_scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )

    # 是否使用 RULER（若不可用会自动回退到相对比较）
    try:
        from art.rewards import ruler_score_group
        use_ruler = True
    except Exception:
        use_ruler = False

    for batch in it:
        print(f"[train] step={batch.step} epoch={batch.epoch}")

        groups = []
        for s in batch.items:
            groups.append(
                art.TrajectoryGroup(
                    wrap_rollout(model, rollout)(model, WebSearchScenario(step=batch.step, scenario=s))
                    for _ in range(training_config["rollouts_per_group"])
                )
            )

        finished = await art.gather_trajectory_groups(
            groups, pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )

        _log_batch_to_wandb(batch=batch, finished_groups=finished, use_ruler=use_ruler)

        if use_ruler:
            extra_litellm_params = {"api_base": "http://localhost:6688", "api_key": os.environ["OPENAI_API_KEY"]}
            judged = []
            for g in finished:
                # RULER 将比较同组候选，给予相对分；无需绝对参考答案
                jg = await ruler_score_group(g, RULER_MODEL, extra_litellm_params=extra_litellm_params, debug=True)
                judged.append(jg)

            await model.train(
                judged,
                config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
                _config={"logprob_calculation_chunk_size": 8},
            )
            wandb.log({"train/used_judged_groups": 1}, step=batch.step)
        else:
            await model.train(
                finished,
                config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
            )
            wandb.log({"train/used_judged_groups": 0}, step=batch.step)

        if batch.step >= training_config["max_steps"]:
            break

    wandb.finish()

if __name__ == "__main__":
    asyncio.run(main())
