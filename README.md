# RLTrainPPT: 基于强化学习的PPT内容填充智能体

本项目旨在训练一个能够自动完成PPT内容研究与填充的AI智能体。该智能体接收一个结构化的PPT大纲（JSON格式），通过强化学习（Reinforcement Learning）进行训练，学会使用网络搜索工具来查找相关信息，填充大纲中的细节内容，并提供引用来源，最终输出一个内容详实、结构完整的PPT内容JSON。

## 核心技术

- **[Agentic RL Transformer (ART)](https://github.com/google-deepmind/art)**: Google DeepMind 开源的强化学习框架，用于训练智能体。本项目使用其 `GRPO` (Generative Rollout Policy Optimization) 算法对模型进行优化。
- **[LangGraph](https://github.com/langchain-ai/langgraph)**: 一个用于构建可循环、有状态的、基于LLM的应用的库。本项目用它来构建 ReAct (Reasoning and Acting) 风格的智能体，使其能够进行思考、行动（搜索）、观察的循环。
- **智谱AI SDK (`zai-sdk`)**: 用于调用智谱的在线网页搜索API，作为智能体获取外部信息的工具。
- **Weights & Biases (`wandb`)**: 用于记录和可视化训练过程中的指标和轨迹，方便监控和分析模型性能。

## 工作流程

项目的核心工作流程如下：

1.  **输入**: 提供一个结构化的PPT大纲JSON。该大纲定义了每页幻灯片的类型（如封面、目录、内容页）和标题，但具体内容是待填充的占位符（例如 `"text": "Detailed content about..."`）。
2.  **智能体处理 (`Rollout`)**:
    -   智能体接收大纲，并根据系统指令（System Prompt）开始工作。
    -   它会遍历大纲中的每一个待填充项。
    -   针对每一项，它会生成一个精确的搜索查询，并调用 `web_search_tool` 工具。
    -   它分析搜索结果，提炼出关键信息，生成2-4句话的文本内容。
    -   在生成的文本末尾，它会附上来源引用标记，如 `[1]`, `[2]`。
    -   它将填充好的内容放回原始JSON结构中，确保结构不被破坏。
3.  **输出**: 智能体调用 `return_filled_outline_tool` 工具，返回一个包含所有填充内容和引用URL列表的完整JSON。
4.  **奖励计算**: 系统会根据输出结果的质量计算奖励分数。奖励函数综合考虑了以下几个方面：
    -   **格式奖励 (`format_reward`)**: 检查输出是否保持了原始结构、文本是否被有效填充、引用标记是否正确使用。
    -   **搜索奖励 (`search_reward`)**: 评估引用来源的多样性、内容是否包含具体数据（如数字、年份）、以及引用标记是否与来源列表一致。
5.  **模型训练**: `art` 框架根据计算出的奖励，使用 `GRPO` 算法对模型进行微调，使其在后续任务中表现得更好。

## 项目结构

```
RLTrainPPT/
├── backend/
│   └── ART_Langgraph_content/
│       ├── train.py         # 核心训练脚本
│       ├── model_test.py    # 用于测试已训练模型的推理能力
│       ├── requirements.txt # Python依赖包
│       └── ...
├── ART/                     # ART 框架（可能作为子模块）
└── README.md                # 本文件
```

## 快速开始

### 1. 环境准备

-   确保您已安装 Python 3.10 或更高版本。
-   一个能够调用智谱API的密钥。

### 2. 安装

```bash
# 1. 克隆仓库
git clone https://github.com/johnson7788/RLTrainPPT

cd RLTrainPPT

# 2. 进入后端代码目录
cd backend/ART_Langgraph_content

# 3. (推荐) 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 4. 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量

在 `backend/ART_Langgraph_content` 目录下创建一个 `.env` 文件，并填入必要的密钥信息。
```
cp env_template .env
```

### 4. 运行项目

#### 训练模型

执行训练脚本来启动智能体的训练过程。训练指标和结果将被记录到 Weights & Biases。

```bash
python LLM_cache.py   #大模型缓存代理，用于请求Openai
python train.py
```

#### 测试模型

训练完成后，可以执行测试脚本来验证模型是否能根据指令（例如：“Who is the CFO of Tesla?”）正确调用工具并返回结果。

```bash
python model_test.py
```
