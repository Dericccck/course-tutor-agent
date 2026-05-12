# course-tutor-agent

一个基于本地课程资料构建的课程辅导 Agent 项目。

项目目标是围绕当前仓库中的 AI 学习内容，逐步实现一个不依赖 `LangChain`、`LangGraph` 的实战型 Agent。当前阶段先从最小可运行版本开始，只完成“资料加载 + 本地检索”这条基础链路，后续再继续补齐问答、总结和学习路径建议。

## 项目背景

这个项目主要面向当前仓库中的课程资料，例如：
- `1-3-SafeandReliableAIviaGuardrails`
- `2-1-RetrievalAugmentedGeneration`
- `2-2-BuildingAndEvaluatingAdvancedRAGApplications`
- `2-3-ai-agents-for-beginners`

希望最终把这些课程内容组织成一个可以辅助学习的 Agent，支持：
- 回答课程相关问题
- 总结某个 notebook / 某节课
- 根据学习目标给出学习顺序建议

## 当前版本已实现功能

目前仓库中的代码已经完成以下能力：

1. 课程资料扫描与读取
- 递归扫描本地课程目录中的 Markdown 文件
- 自动过滤 `venv`、`.venv`、`site-packages`、`__pycache__`、`.git`、`node_modules` 等无关目录
- 将读取到的课程资料统一转换为结构化 `Document`

2. 文档基础结构化
- 为每份资料提取：
  - `source`
  - `title`
  - `content`
  - `doc_type`
  - `tags`
- 优先使用 Markdown 标题作为文档标题
- 根据路径粗粒度打上 `agent`、`rag`、`guardrails` 等标签

3. 本地关键词检索
- 对用户问题做简单 token 拆分
- 根据“标题 / 标签 / 正文”命中情况进行打分
- 对 `notebook-summary.md` 这类课程摘要文档给予更高权重
- 返回前 `top_k` 个最相关结果

4. 检索结果可解释输出
- 返回检索分数
- 返回来源路径
- 返回命中的文本片段 `snippet`

## 当前版本未完成的部分

下面这些能力已经在目录中预留，但还没有正式接上：
- `app/agent.py`：主 Agent 编排流程
- `app/prompts.py`：提示词模板
- `app/config.py`：配置集中管理
- 基于大模型的最终回答生成
- notebook / 节次总结
- 基于用户目标的学习路径建议

## 当前目录结构

```text
course-tutor-agent/
  app/
    agent.py
    config.py
    loader.py
    main.py
    prompts.py
    retriever.py
    schemas.py
  data/
  tests/
    test_loader.py
```

## 目前可运行的内容

当前可以直接运行 `main.py`，验证“资料加载 + 检索”是否正常工作：

```bash
cd /Users/a1-6/Desktop/AIAgent/05-project/course-tutor-agent/app
python main.py
```

运行后会：
- 加载本地课程 Markdown 文档
- 使用一个测试问题进行检索
- 输出命中的课程资料标题、分数、来源和摘要片段

## 环境变量

项目后续会同时支持两种模型接入方式：
- `OpenAI API`
- `GitHub Models`（通过 `GITHUB_TOKEN` 和 OpenAI 兼容接口）

示例配置见 `.env.example`：

```env
LLM_PROVIDER=github
MODEL_NAME=gpt-4.1-mini
OPENAI_API_KEY=
OPENAI_BASE_URL=
GITHUB_TOKEN=
GITHUB_MODELS_BASE_URL=https://models.inference.ai.azure.com/
COURSE_SOURCE_ROOT=/Users/a1-6/Desktop/AIAgent/code
RETRIEVAL_TOP_K=5
```

## 下一步计划

接下来会按以下顺序继续完善：

1. 接入提示词与主流程编排
2. 把检索结果送给大模型生成回答
3. 支持“课程问答”
4. 支持“某节课 / 某个 notebook 总结”
5. 支持“根据学习目标生成学习顺序建议”
