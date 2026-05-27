# course-tutor-agent

一个基于本地课程资料构建的课程辅导 Agent 项目。

项目目标是围绕当前仓库中的 AI 学习内容，逐步实现一个不依赖 `LangChain`、`LangGraph` 的实战型 Agent。当前已经完成 `v1` 的可交互 CLI 主链路，并进入 `v2`：逐步把检索层从整篇文档检索升级为 `chunk`、`vector` 和 `hybrid` 检索。

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
- 将整篇文档按 Markdown 标题优先切分为多个 `chunk`
- 新增 chunk 级检索与 snippet 提取
- 为同一 source 增加结果去重限制，避免单篇文档占满结果位
- 支持通过 `RETRIEVAL_MODE` 在 `document` / `chunk` 检索之间切换
- 在主流程中优先使用 chunk 检索，并保留整篇文档检索作为 fallback
- 为 chunk 检索和 chunk 主流程接入补充测试

## V2 Phase 2

当前 `v2` 第二阶段已完成以下升级：
- 新增 `embedding_provider.py`，拆分 embedding provider 相关职责
- 新增 `EmbeddingProvider` 协议与 `NotImplementedEmbeddingProvider`
- 新增 `build_embedding_text(...)`，统一构造 embedding 输入文本
- 新增 `HashEmbeddingProvider` 作为本地可运行的过渡实现
- 新增 `SentenceTransformerEmbeddingProvider`，支持接入真实 embedding 模型
- 新增 `VectorStore` 协议、`FakeVectorStore` 与 `InMemoryVectorStore`
- 支持通过 `EMBEDDING_PROVIDER`、`EMBEDDING_MODEL_NAME`、`EMBEDDING_CACHE_DIR` 配置真实 embedding provider
- `RETRIEVAL_MODE=vector` 已打通到主流程，并可结合 `sentence-transformers` 进行真实向量检索
- 新增 `RETRIEVAL_MODE=hybrid`，支持词法检索与向量检索混合召回
- 新增 `reranker.py`，支持在 `hybrid` 结果上使用本地 CrossEncoder 做二次重排
- 支持通过 `RERANKER_PROVIDER`、`RERANKER_MODEL_NAME`、`RERANKER_CACHE_DIR` 配置本地 reranker
- `study_plan` 场景新增 post-rank 规则层，让学习路线服从课程教学主线
- 新增本地向量索引缓存，避免每次启动都重新计算全部 chunk embeddings
- 增加课程目录白名单和文件级白名单，进一步收紧索引语料范围
- 为 vector provider / store 与 hybrid 检索补充测试

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
- 支持通过课程目录白名单收紧扫描范围
- 支持通过文件级白名单优先只索引 `notebook-summary.md` 等核心资料
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
- 优先按 Markdown 标题边界切分 section
- section 过长时，再退回按段落和长度累计切块
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
- 支持通过配置切换 `document` / `chunk` / `vector` / `hybrid` 四种检索模式

5. Prompt 组装
- 将检索结果组织成模型可读的上下文块
- 构造系统提示词，约束模型只能基于课程资料回答
- 明确要求模型输出 JSON 结构

6. LLM 问答主链路
- 根据配置创建模型客户端
- 支持 `OpenAI API` 和 `GitHub Models`
- 将“用户问题 + 检索结果”发送给模型
- 获取课程问答结果
- 当前主流程默认优先使用 chunk 检索结果
- 可通过 `RETRIEVAL_MODE` 切回整篇文档检索
- 保留整篇文档检索作为 fallback
- `RETRIEVAL_MODE=vector` 时已支持注入 `vector_store`
- `RETRIEVAL_MODE=hybrid` 时会合并 chunk 检索与向量检索结果
- `RETRIEVAL_MODE=hybrid` 时会先宽召回，再按优先级与同源限制合并结果

7. 向量检索链路
- 支持通过 `build_embedding_text(...)` 统一构造 embedding 输入
- 支持 `HashEmbeddingProvider` 作为本地快速验证用的 embedding provider
- 支持 `SentenceTransformerEmbeddingProvider` 接入真实 embedding 模型
- 支持通过 `EMBEDDING_CACHE_DIR` 控制模型缓存目录
- 当前 embedding 模型与 reranker 模型都支持本地优先加载，不依赖运行时重新下载
- `InMemoryVectorStore` 已支持：
  - `index_chunks(...)`
  - `load_index(...)`
  - `search(...)`
  - 基于简单点积的最小可运行向量检索

8. Hybrid 重排与学习路线后排序
- `RETRIEVAL_MODE=hybrid` 时，主流程会先执行：
  - lexical recall
  - vector recall
  - merge 去重
- 检索相关配置已经在代码层收口为 `RetrievalSettings`
- 当前会统一通过 `settings.retrieval.xxx` 读取以下配置：
  - `retrieval_top_k`
  - `retrieval_mode`
  - `summary_strategy`
  - `hybrid_candidate_multiplier`
  - `hybrid_candidate_minimum`
  - `embedding_provider`
  - `embedding_model_name`
  - `embedding_cache_dir`
  - `reranker_provider`
  - `reranker_model_name`
  - `reranker_cache_dir`
- `HYBRID_CANDIDATE_MULTIPLIER` 和 `HYBRID_CANDIDATE_MINIMUM` 用于控制 hybrid 模式下的候选池规模
- 当前 `candidate_top_k` 的计算方式为：
  - `max(RETRIEVAL_TOP_K * HYBRID_CANDIDATE_MULTIPLIER, HYBRID_CANDIDATE_MINIMUM)`
- 如果启用了 reranker，还会继续执行：
  - CrossEncoder rerank
- 对 `study_plan` 问题，还会额外执行 post-rank：
  - 普通学习路线优先按课程主线排序
  - `RAG -> Agentic RAG` 路线优先把 `2-2/L1`、`2-2/L2` 排到 `05 Agentic RAG` 前面
- `study_plan` 的后排序规则已从代码硬编码升级为配置驱动
- 学习路线顺序配置文件位于 `data/study_plan_order.json`
- `default_title_order` 用于定义普通学习路线的课程主线顺序
- `rag_route_priorities` 用于定义 `RAG -> Agentic RAG` 路线的优先级规则
- 后续如果要调整学习顺序，不再需要改 `agent.py`，只需要改配置文件
- `SUMMARY_STRATEGY` 用于控制 summary 场景的结果收窄策略
- 当前仅支持：
  - `same-source`
  - 只保留与首条结果来自同一 source 的检索结果，降低跨文档污染

9. 检索与答案评估
- 新增 `eval/questions.json` 作为评估问题集
- 新增 `eval/run_eval.py` 作为本地评估脚本
- 当前评估分为两层：
  - Retrieval Eval：看 source 命中与课程簇覆盖
  - Agent Eval：看真实 `ask_course_agent(...)` 输出
- `eval/run_eval.py` 支持通过环境变量控制评估范围：
  - `RUN_AGENT_EVAL=false`
  - 只运行 Retrieval Eval，跳过真实模型调用
  - `RUN_AGENT_EVAL=true`
  - 同时运行 Retrieval Eval 和 Agent Eval
  - `EVAL_MODES=chunk,vector,hybrid`
  - 控制本次评估运行哪些检索模式，也可以只跑 `hybrid` 或 `chunk,hybrid`
- 当前默认行为：
  - `RUN_AGENT_EVAL` 未设置时按 `false` 处理
  - `EVAL_MODES` 未设置时按 `chunk,vector,hybrid` 处理
- `eval/questions.json` 里的单条样本还支持通过 `enabled_modes` 做模式级过滤：
  - 未设置 `enabled_modes` 时，默认参与所有 mode 的评估
  - 设置后，只会在指定 mode 下参与评估
  - 例如：`"enabled_modes": ["chunk", "hybrid"]`
- `eval/questions.json` 里的单条样本还支持通过 `tags` 做标签分组：
  - 例如：`"tags": ["study_plan", "rag"]`
  - 方便只跑某一类评估题，而不必每次都跑整套样本
- `eval/run_eval.py` 支持通过 `EVAL_TAGS` 筛选样本标签：
  - `EVAL_TAGS=study_plan`
  - 只跑带 `study_plan` 标签的样本
  - `EVAL_TAGS=rag,study_plan`
  - 只跑同时命中 `rag` 或 `study_plan` 标签的样本集合
- 示例：
  - `RUN_AGENT_EVAL=false /Users/a1-6/Desktop/AIAgent/05-project/.conda/bin/python eval/run_eval.py`
  - `RUN_AGENT_EVAL=true EVAL_MODES=hybrid /Users/a1-6/Desktop/AIAgent/05-project/.conda/bin/python eval/run_eval.py`
  - `RUN_AGENT_EVAL=true EVAL_MODES=hybrid EVAL_TAGS=study_plan /Users/a1-6/Desktop/AIAgent/05-project/.conda/bin/python eval/run_eval.py`
- 对 `study_plan` 场景，主评估指标为：
  - `Expected Answer Hit`
  - `Forbidden Answer Hit`
- `Primary Hit` 在带 memory 的学习路线问题中仅作为参考，不再是主指标

10. 向量索引持久化
- 支持通过 `VECTOR_INDEX_CACHE_PATH` 配置本地向量索引缓存文件
- 首次运行时：
  - 计算全部 chunk embeddings
  - 自动生成 `vector_index_cache.json`
- 后续运行时：
  - 如果 embedding 配置、课程范围和 chunk 指纹都一致，则直接从缓存恢复索引
  - 如果任一条件变化，则自动重建并覆盖缓存
- 当前缓存解决的是：
  - embedding 计算与索引构建成本
- 当前缓存不解决的是：
  - 模型本身每次进程启动时的内存加载成本

11. 任务分流
- 支持按问题类型区分：
  - 普通课程问答
  - 某节课 / 某个 notebook 总结
  - 学习顺序建议
- 针对不同任务使用不同的 prompt 组织方式

12. 命令行连续交互
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

13. 本地用户记忆
- 使用本地 `JSON` 文件保存用户记忆
- 当前支持记录：
  - `learning_goal`
  - `preferred_scope`
  - `completed_topics`
- 在回答时把用户目标和学习范围一并带入 prompt
- 在执行章节总结后自动记录已完成主题
- 在生成学习顺序建议时，优先参考未完成模块并尽量避开重复推荐已完成主题
- 在生成学习顺序建议时，优先遵守用户设置的学习范围，并将范围外内容降级为补充建议

14. 结构化输出
- 将模型输出解析为统一的 `AgentAnswer`
- 返回：
  - `answer`
  - `suggestions`
  - `sources`
- 当模型输出不规范时，提供兜底处理，避免程序直接崩溃

15. 最小自动化测试
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
- 覆盖配置校验行为：
  - `RETRIEVAL_MODE=chunk`
  - `RETRIEVAL_MODE=document`
  - `RETRIEVAL_MODE=vector`
  - `RETRIEVAL_MODE=hybrid`
  - 非法检索模式值被明确拒绝
  - 非法 `EMBEDDING_PROVIDER` 值被明确拒绝
- 覆盖 chunk 检索行为：
  - chunk 数量生成
  - 总结问题命中目标章节 chunk
  - Tool Use 问题命中目标模块 chunk
  - 同一 source 的 chunk 去重限制
  - chunk snippet 非空
  - chunk 检索结果携带真实 `chunk_id`
- 覆盖 vector provider / store 骨架行为：
  - `build_embedding_text(...)` 包含标题、标签和正文
  - 空标签时稳定输出“无”
  - `NotImplementedEmbeddingProvider` 明确抛出未实现异常
  - `FakeVectorStore` 可保存 chunks 并返回预设结果
  - `InMemoryVectorStore.index_chunks(...)` 会调用 embedding provider
  - `InMemoryVectorStore.search(...)` 会返回排序后的结果
- 覆盖切块策略行为：
  - Markdown 内容按标题边界切 section
  - chunk 优先保留 section 边界
  - 超长 section 退回按段落切块
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
  - `RETRIEVAL_MODE=document` 时强制走 document 检索
  - `RETRIEVAL_MODE=vector` 时如果没传 `vector_store` 会明确报错
  - `RETRIEVAL_MODE=vector` 时会优先使用 `vector_store.search(...)`
  - `merge_retrieval_results(...)` 会按优先级合并并去重
  - `RETRIEVAL_MODE=hybrid` 时会同时使用 chunk 检索和向量检索
  - `RETRIEVAL_MODE=hybrid` 时如果缺少 `vector_store` 会明确报错
  - `RETRIEVAL_MODE=hybrid` 时如果启用了 reranker，会对合并后的候选结果继续重排
  - summary 场景会收窄到首条命中的目标文档来源，避免跨文档稀释总结
  - summary 场景在 chunk 路径下也会更新 completed_topics
  - `post_rank_study_plan_results(...)` 会对学习路线结果按课程主线做最终后排序
  - `RAG -> Agentic RAG` 路线会优先输出 `L1 -> L2 -> 05 Agentic RAG`
  - `InMemoryVectorStore.load_index(...)` 可直接从缓存恢复 chunks 和 embeddings
- 覆盖向量索引缓存行为：
  - `build_chunk_fingerprint(...)` 的稳定性
  - chunk 内容变化时指纹变化
  - `is_vector_index_cache_valid(...)` 的命中与失效判断
  - 向量索引可从缓存直接恢复
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
- 更高质量的向量索引与相似度计算
- 更正式的 hybrid 重排策略
- 更精细的 chunk 切分与引用控制
- 更智能的长期记忆与学习进度利用
- 更丰富的 CLI 交互体验（如菜单、历史记录、参数模式）

## 当前目录结构

```text
course-tutor-agent/
  app/
    agent.py
    config.py
    embedding_provider.py
    loader.py
    main.py
    memory.py
    prompts.py
    retriever.py
    schemas.py
    vector_store.py
  data/
    user_memory.json
  tests/
    conftest.py
    test_agent.py
    test_chunk_retriever.py
    test_config.py
    test_loader.py
    test_main_helpers.py
    test_memory.py
    test_prompts.py
    test_retriever.py
    test_study_plan_memory.py
    test_task_routing.py
    test_vector_store.py
```

## 目前可运行的内容

当前可以直接运行 `main.py`，启动课程辅导 Agent 的交互式 CLI：

```bash
cd /Users/a1-6/Desktop/AIAgent/05-project/course-tutor-agent
/Users/a1-6/Desktop/AIAgent/05-project/.conda/bin/python app/main.py
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
HYBRID_CANDIDATE_MULTIPLIER=3
HYBRID_CANDIDATE_MINIMUM=10
SUMMARY_STRATEGY=same-source
RETRIEVAL_MODE=chunk
```

本地实际运行时，请在项目根目录创建 `.env`，并填写真实 key / token。  
`.env` 不应提交到 GitHub。

## 下一步计划

接下来会按以下顺序继续完善：

1. 继续提升 chunk 检索质量与引用精度
2. 接入第一个真实 embedding provider，升级为更正式的向量检索
3. 扩展测试覆盖到更多实际问答场景
4. 继续完善学习路径建议和上下文管理
5. 逐步增强长期记忆与学习进度利用
