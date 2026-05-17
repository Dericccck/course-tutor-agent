# course-tutor-agent

一个基于本地课程资料构建的课程辅导 Agent 项目。

项目目标是围绕当前仓库中的 AI 学习内容，逐步实现一个不依赖 `LangChain`、`LangGraph` 的实战型 Agent。当前已经完成 `v1` 的可交互 CLI 主链路，并开始进入 `v2` 的第一阶段：将检索层从整篇文档检索升级为基于 `chunk` 的检索。

## V1 Scope

当前 `v1` 版本已经覆盖以下核心范围：
- 基于本地课程资料的课程问答
- 某节课 / 某个 notebook 的课程总结
- 基于学习目标、学习范围和已完成主题的学习顺序建议
- 本地学习记忆管理与学习进度摘要
- 可交互 CLI 入口
- 面向本地逻辑的基础自动化测试

## V2 Phase 1

当前 `v2` 第一阶段已完成以下升级：
- 为 Markdown 课程资料新增 `DocumentChunk` 数据结构
- 将整篇文档切分为多个 `chunk`
- 新增 chunk 级检索与 snippet 提取
- 为同一 source 增加结果去重限制，避免单篇文档占满结果位
- 在主流程中优先使用 chunk 检索，保留整篇文档检索作为 fallback
- 为 chunk 检索和 chunk 主流程接入补充测试

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
- 支持将单篇 `Document` 切分为多个 `DocumentChunk`

3. chunk 级检索准备
- 按段落和长度累计对 Markdown 文档进行切块
- 为每个 chunk 生成独立的 `chunk_id`
- 支持批量加载全部文档对应的 chunks

4. 本地关键词检索
- 对用户问题做简单 token 拆分
- 根据“标题 / 标签 / 正文”命中情况进行打分
- 支持对问题中的短语与文档标题做更直接的匹配加分
- 对 `notebook-summary.md` 这类课程摘要文档给予更高权重
- 返回前 `top_k` 个最相关结果
- 支持基于 `DocumentChunk` 的 chunk 级检索
- 对同一 `source` 的检索结果增加数量限制，减少单篇文档重复占位
- 检索结果支持携带 `chunk_id`，为后续更细粒度引用做准备

5. Prompt 组装
- 将检索结果组织成模型可读的上下文块
- 构造系统提示词，约束模型只能基于课程资料回答
- 明确要求模型输出 JSON 结构

6. LLM 问答主链路
- 根据配置创建模型客户端
- 支持 `OpenAI API` 和 `GitHub Models`
- 将“用户问题 + 检索结果”发送给模型
- 获取课程问答结果
- 当前主流程优先使用 chunk 检索结果
- 保留整篇文档检索作为 fallback

7. 任务分流
- 支持按问题类型区分：
  - 普通课程问答
  - 某节课 / 某个 notebook 总结
  - 学习顺序建议
- 针对不同任务使用不同的 prompt 组织方式

8. 命令行连续交互
- 程序启动后只加载一次课程资料
- 支持连续输入多个问题
- 支持 `help` 查看帮助
- 支持 `set_goal:` 设置学习目标
- 支持 `set_scope:` 设置学习范围
- 支持 `mark_done:` 手动标记已完成主题
- 支持 `show_memory` 查看当前学习记忆
- 支持 `show_progress` 查看用户友好的学习进度摘要
- 支持 `show_examples` 重新查看示例问题
- 支持 `clear_goal` 清除学习目标
- 支持 `clear_scope` 清除学习范围
- 支持 `clear_done` 清空所有已完成主题
- 支持 `reset_memory` 重置全部学习记忆
- 支持 `unmark_done:` 取消某个已完成主题
- 支持通过 `exit` / `quit` / `q` 退出

9. 本地用户记忆
- 使用本地 `JSON` 文件保存用户记忆
- 当前支持记录：
  - `learning_goal`
  - `preferred_scope`
  - `completed_topics`
- 在回答时把用户目标和学习范围一并带入 prompt
- 在执行章节总结后自动记录已完成主题
- 在生成学习顺序建议时，优先参考未完成模块并尽量避开重复推荐已完成主题
- 在生成学习顺序建议时，优先遵守用户设置的学习范围，并将范围外内容降级为补充建议

10. 结构化输出
- 将模型输出解析为统一的 `AgentAnswer`
- 返回：
  - `answer`
  - `suggestions`
  - `sources`
- 当模型输出不规范时，提供兜底处理，避免程序直接崩溃

11. 最小自动化测试
- 已添加基础 pytest 测试
- 覆盖任务分流逻辑：
  - `qa`
  - `summary`
  - `study_plan`
- 覆盖关键检索行为：
  - Tool Use 问答命中
  - 章节总结命中
  - 学习路线问题命中 Agent 相关资料
  - 标题短语匹配后的稳定命中
  - 整篇文档检索路径下 `chunk_id` 为 `None`
- 覆盖 chunk 检索行为：
  - chunk 数量生成
  - 总结问题命中目标章节 chunk
  - Tool Use 问题命中目标模块 chunk
  - 同一 source 的 chunk 去重限制
  - chunk snippet 非空
  - chunk 检索结果携带真实 `chunk_id`
- 覆盖本地用户记忆读写：
  - 默认结构返回
  - 保存与读取一致性
  - 自动创建 memory 文件目录
- 覆盖 prompt 组织逻辑：
  - memory 展开
  - 普通问答 prompt 拼接
  - 学习路线 prompt 中已完成 / 未完成模块划分
- 覆盖 agent 主流程本地行为：
  - 无检索结果时的兜底回答
  - summary 任务自动更新 completed_topics
  - 模型返回非法 JSON 时的 fallback 处理
  - 传入 chunks 时优先走 chunk 检索
  - summary 场景在 chunk 路径下也会更新 completed_topics
- 覆盖学习进度与学习路线联动：
  - 已完成模块与未完成模块划分
  - 学习目标 / 学习范围 / 已完成主题共同进入学习路线 prompt
  - 学习范围对主路线与补充建议的约束
- 覆盖 CLI 辅助输出：
  - help 文本包含关键命令
  - 示例问题覆盖问答 / 总结 / 学习顺序建议三类场景
  - 学习进度摘要在空状态和有进度状态下的正确展示
  - 学习进度摘要中的下一步推荐主题

## 当前版本未完成的部分

下面这些能力还没有正式完成：
- 更正式的向量化 RAG 检索
- 更精细的 chunk 切分与引用控制
- 更智能的长期记忆与学习进度利用
- 更丰富的 CLI 交互体验（如菜单、历史记录、参数模式）

## 当前目录结构

```text
course-tutor-agent/
  app/
    agent.py
    config.py
    loader.py
    main.py
    memory.py
    prompts.py
    retriever.py
    schemas.py
  data/
    user_memory.json
  tests/
    conftest.py
    test_agent.py
    test_chunk_retriever.py
    test_loader.py
    test_main_helpers.py
    test_memory.py
    test_prompts.py
    test_retriever.py
    test_study_plan_memory.py
    test_task_routing.py
```

## 目前可运行的内容

当前可以直接运行 `main.py`，启动课程辅导 Agent 的交互式 CLI：

```bash
cd /Users/a1-6/Desktop/AIAgent/05-project/course-tutor-agent/app
python main.py
```

运行后会：
- 加载本地课程 Markdown 文档
- 连续接收用户输入的问题
- 根据问题类型执行课程问答、章节总结或学习顺序建议
- 支持设置学习目标和学习范围
- 支持自动记录已总结过的主题
- 支持手动维护已完成主题和查看当前学习记忆
- 支持查看用户友好的学习进度摘要
- 支持根据当前完成情况给出下一步优先学习主题
- 输出回答、学习建议和引用来源
- 支持输入 `exit` / `quit` / `q` 退出

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

本地实际运行时，请在项目根目录创建 `.env`，并填写真实 key / token。  
`.env` 不应提交到 GitHub。

## 下一步计划

接下来会按以下顺序继续完善：

1. 继续提升 chunk 检索质量与引用精度
2. 视需要引入 embedding / 向量检索，升级为更正式的 RAG
3. 扩展测试覆盖到更多实际问答场景
4. 继续完善学习路径建议和上下文管理
5. 逐步增强长期记忆与学习进度利用
