# TalentMatch AI

> 合规优先的 AI 招聘 Sourcing Copilot MVP：从 JD 自动生成搜索策略，发现候选人线索，完成证据级匹配评分，并辅助 HR 生成人工确认后的触达草稿。

TalentMatch AI 是一个面向 HR、猎头、招聘运营和早期团队的 AI 招聘工作台 Demo。它不是“自动抓取招聘平台”的工具，而是把 **岗位理解、候选人检索、匹配判断、触达草稿和招聘 CRM** 组织成一个可解释、可审计、可演示的完整闭环。

当前版本基于 Streamlit 实现，内置 100 条 mock 候选人数据，支持 DeepSeek API 模式与 Mock Demo 模式，适合用于产品面试、AI 应用作品集、招聘自动化原型验证和技术方案讨论。

## 产品背景

招聘工作中最耗时的部分往往不是“联系候选人”，而是联系之前的判断：

- JD 信息不结构化，HR 需要手动拆解岗位要求。
- 招聘平台、人才库、表格里的候选人信息分散，筛选成本高。
- 简历关键词与真实经历表达不一致，容易漏掉合适候选人。
- 传统表格只能展示信息，不能解释“为什么推荐这个人”。
- 触达话术容易模板化，缺少候选人个性化依据。
- 合规边界、黑名单、不再联系、审计记录经常被流程忽略。

TalentMatch AI 的目标是成为一个招聘 Copilot：让系统先完成资料整理、候选人召回、证据级分析和草稿准备，再由招聘人员做最终判断。

## 核心能力

| 模块 | 能力说明 |
| --- | --- |
| JD 解析 | 将非结构化岗位描述解析为岗位名称、地点、职责、硬性要求、加分项、排除项和评分权重。 |
| 自动检索 | 根据 JD 生成关键词、职位别名、城市、行业、排除词和布尔搜索式。 |
| 候选人召回 | 从 mock source、本地人才库、CSV / Excel、手动粘贴资料中召回候选人线索。 |
| 100 人 mock 候选池 | 内置高匹配、弱匹配、岗位不匹配、缺失字段、黑名单、不再联系、多来源等样例。 |
| AI 匹配评分 | 输出匹配分、推荐等级、置信度、命中理由、风险点、缺失信息和面试关注点。 |
| 证据级解释 | 每个推荐结果都展示 JD 要求与候选人资料之间的对应证据。 |
| 触达草稿 | 生成专业正式、轻松自然、猎头风格等个性化触达话术。 |
| 招聘 CRM | 管理候选人状态、备注、下一步行动、提醒时间、黑名单、不再联系和 CSV 导出。 |
| 合规审计 | 记录导入、检索、评分、草稿生成、状态变更、删除和导出等关键动作。 |

## 产品亮点

### 1. 自动 Sourcing Copilot

用户输入 JD 后，系统会自动生成搜索策略，并模拟从多种候选人来源召回线索。候选人不会直接进入触达流程，而是先进入 Sourcing Inbox，由 HR 复核后加入短名单。

### 2. DeepSeek / Mock 双模式

如果配置了 `DEEPSEEK_API_KEY`，系统会调用 DeepSeek 进行结构化分析；如果没有 API Key，则自动进入 Mock Demo 模式，保证无外部网络或无密钥时也能完整演示。

### 3. 证据级 AI 判断

TalentMatch AI 不只给一个分数，还会给出：

- 为什么匹配
- 哪些信息不足
- 存在哪些风险
- 面试应该重点追问什么
- 是否适合触达

这让 AI 判断更接近招聘业务决策，而不是黑盒评分。

### 4. 不可触达护栏

黑名单、不再联系候选人可以参与分析评分，但不能进入触达草稿生成流程。系统始终区分“可以分析”和“可以联系”，避免把筛选工具误用成自动骚扰工具。

### 5. 高保真 Streamlit 工作台

当前 UI 已从默认表格 Demo 升级为招聘工作台：

- 今日招聘驾驶舱
- 三步工作流导航
- 候选人卡片
- 自动检索页
- Sourcing Inbox
- 高对比表格
- 来源与授权中心
- CRM 状态管理

## 系统工作流

```text
JD 输入
  ↓
JD 结构化解析
  ↓
搜索策略生成
  ↓
候选人召回
  ↓
去重与资料归一
  ↓
AI / Mock 匹配评分
  ↓
证据级解释与排序
  ↓
触达草稿生成
  ↓
HR 人工确认
  ↓
招聘 CRM 跟进
```

## 技术架构

```text
talentmatch-ai/
├── app.py                  # Streamlit 主应用：页面、状态、工作流与 UI
├── llm_client.py           # DeepSeek API wrapper、JSON 解析、mock fallback
├── sourcing.py             # 搜索策略生成、mock search、本地候选人召回
├── utils.py                # 候选人去重、评分辅助、状态更新、合规检查、导出
├── maimai_verifier.py      # 平台候选人公开信息验证实验模块
├── sample_candidates.csv   # 100 条 mock 候选人数据
├── requirements.txt        # Python 依赖
├── .env.example            # API Key 示例配置
└── tests/                  # 核心逻辑测试
```

核心技术栈：

- **Streamlit**：快速搭建可交互 MVP。
- **pandas**：候选人数据处理、CSV / Excel 读取与导出。
- **DeepSeek Chat API**：JD 解析、匹配分析、触达草稿生成。
- **Mock fallback**：无 API Key 时保证 Demo 可运行。
- **规则 + 结构化评分**：敏感词检测、黑名单护栏、资料完整度、来源可信度。
- **unittest**：核心逻辑、样例数据和验证模块测试。

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/<your-username>/talentmatch-ai.git
cd talentmatch-ai
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动应用

```bash
streamlit run app.py
```

启动后浏览器会打开本地 Streamlit 页面。首次进入时，系统会自动载入 100 条 mock 候选人数据。

## DeepSeek API Key 配置

方式一：使用环境变量。

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
streamlit run app.py
```

方式二：复制 `.env.example` 为 `.env`。

```bash
cp .env.example .env
```

然后填写：

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
```

方式三：启动应用后，在 Streamlit 侧边栏输入 API Key。

项目不会展示、打印、导出或持久化保存你的 API Key。

## Mock Demo 模式

如果没有检测到 `DEEPSEEK_API_KEY`，系统会自动进入 Mock Demo 模式。Mock 模式仍然可以完整跑通：

```text
JD 输入 → 自动检索 → 候选人召回 → AI/Mock 评分 → 候选人排序 → 草稿生成 → CRM 管理 → CSV 导出
```

这让项目在面试、路演或无外部网络环境中仍然稳定可演示。

## 3-5 分钟 Demo 路径

1. 进入 `1 定义岗位`，点击 `填充示例 JD`。
2. 点击 `AI 解析 JD`，查看结构化岗位画像并保存岗位。
3. 进入 `0 自动检索`，查看系统生成的关键词、职位别名和布尔搜索式。
4. 点击 `运行 Mock 自动检索`，在 Sourcing Inbox 查看新线索。
5. 将合适线索加入短名单。
6. 进入 `3 评估候选人`，默认对 100 人候选池进行匹配分析。
7. 查看候选人卡片中的匹配分、置信度、资料完整度、来源可信度、风险和下一步动作。
8. 进入 `候选人详情`，生成个性化触达草稿。
9. 保存为 `待人工确认`。
10. 进入 `确认触达`，更新 CRM 状态、备注、提醒时间并导出 CSV。

## 合规声明

TalentMatch AI 是一个合规优先的招聘效率工具 Demo。

当前项目：

- 不实现非官方爬虫。
- 不绕过招聘平台登录、验证码、风控或权限系统。
- 不自动群发消息。
- 不自动发送触达内容。
- 不保存真实 API Key。
- 不基于敏感个人属性进行评分或推荐。
- 不包含真实候选人隐私数据。

候选人资料默认来自用户主动导入、CSV / Excel、公开信息摘要、企业授权人才库、官方 API 占位或 mock 示例数据。所有触达内容仅生成草稿，最终发送动作应由招聘人员在官方 App、授权渠道或企业内部系统中人工确认。

## 未来路线

### V3：招聘平台 Connector

- 抽象统一 Source Connector。
- 支持官方 API、ATS、企业授权人才库、招聘网站导出数据。
- 引入浏览器分享扩展或系统分享入口，只接收用户主动提供的页面资料。
- 对外部来源记录授权状态、来源链接、抓取时间和字段完整度。

### V4：自动 Sourcing Agent

- 输入 JD 后自动创建 Sourcing Task。
- 自动选择数据源、生成查询、召回候选人并排序。
- 加入任务状态、失败重试、限流、增量刷新和多岗位管理。
- 将候选人反馈、短名单、回复、面试结果回流到排序模型。

### 模型升级方向

- **Embedding 召回**：用向量检索解决关键词不一致的问题。
- **Reranker 重排**：对 Top-K 候选人做更精细排序。
- **逻辑回归 / LightGBM**：基于 HR 反馈训练可解释的短名单概率、回复概率和排序分。
- **LLM 深度解释**：只对高潜候选人做证据级分析和触达文案生成。

### 工程化方向

- 从 Streamlit 迁移到 Next.js / React 前端。
- 使用 FastAPI 提供后端服务。
- 使用 PostgreSQL + pgvector 管理结构化数据和向量索引。
- 使用任务队列处理批量 sourcing。
- 增加团队权限、数据加密、审计后台和招聘漏斗分析。

## 测试

运行核心测试：

```bash
python3 -B -m unittest tests/test_core.py
python3 -B -m unittest tests/test_maimai_verifier.py
```

验证样例候选人数量：

```bash
python3 -B -c "import app; print(len(app.load_sample_candidates()))"
```

预期输出：

```text
100
```

语法检查：

```bash
python3 -m compileall .
```

## 适合展示的项目价值

TalentMatch AI 的重点不是“自动联系更多人”，而是把招聘判断产品化：

- 系统自动拆解岗位。
- 系统自动生成搜索策略。
- 系统自动召回候选人线索。
- AI 给出可解释的匹配证据。
- HR 保留最终判断和触达确认权。
- CRM 记录招聘过程中的每一步。

这让项目既能展示 AI Agent 产品能力，也能展示对招聘业务、数据合规和端到端产品闭环的理解。

## License

This project is for demo and portfolio purposes. Please review compliance requirements before adapting it to production recruiting workflows.
