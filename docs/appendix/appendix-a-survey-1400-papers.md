# 附录A：1400+论文学术综述

> **论文标题**：A Survey of Context Engineering for Large Language Models
> **论文链接**：https://arxiv.org/abs/2507.13334
> **研究团队**：中科院计算技术研究所
> **配套资源**：https://github.com/Meirtz/Awesome-Context-Engineering
> **Hugging Face**：https://huggingface.co/papers/2507.13334

---

## 概述

本工作由中科院计算技术研究所团队主导完成，对超过**1400篇研究论文**进行了系统性分析，首次对LLMs的上下文工程进行了全面和系统的回顾，旨在为未来的上下文感知智能体系统提供清晰的理论基础与系统蓝图。

![综述概览](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDEOs7jjHFG4dlfw38NctrmQsodTyZsqwPicZOkcC66hWuDicYmBwo5lZO9FRKwiadpNWmMDw0shbzaibA/640?wx_fmt=png&from=appmsg&randomid=x22n6th9&tp=wxpic&wxfrom=5&wx_lazy=1)

---

## 1. 研究动机与背景

![研究背景](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGhKg9nnSz5qQrwKvXibt3wulOVRfC18yCkd6xXqGq22h6QUk8chptF0fnQ4uXeZtAktYMrWwG2SyQ/640?wx_fmt=png&randomid=isdmrlvw&tp=wxpic&wxfrom=5&wx_lazy=1)

### 1.1 核心问题

大型语言模型（LLMs）的性能从根本上取决于其在推理过程中获得的上下文信息。随着LLMs从简单的指令遵循系统发展为复杂应用的推理核心，**如何设计和管理其信息有效载荷已演变为一门正式的学科**。

### 1.2 提示工程的局限

传统的"提示工程"（Prompt Engineering）概念已不足以涵盖现代AI系统所需的信息设计、管理和优化的全部范围。这些系统处理的不再是单一、静态的文本字符串，而是一个**动态、结构化且多方面的信息流**。

### 1.3 上下文工程的出现

上下文工程（Context Engineering）的出现，旨在超越简单的提示设计，**系统性地优化供给LLMs的信息有效载荷**。

### 1.4 研究现状与挑战

然而，上下文工程领域的研究虽然发展迅速，却呈现出**高度专业化和碎片化**的特点：

- 现有研究大多孤立地探讨特定技术（RAG、Agent系统、长上下文处理等）
- 缺乏一个统一的框架来系统地组织这些多样化的技术
- 需要阐明技术之间的内在联系

为了应对这一挑战，本综述对超过1400篇研究论文进行了系统性分析，旨在为研究人员和工程师提供一个**清晰的技术路线图**，促进对该领域的深入理解，催化技术创新。

![论文数量统计](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDEOs7jjHFG4dlfw38NctrmQNZicsv0uUDoAib4OFbPRuHcCT5DokVQibzXYVabZAO4lzJtbDtRga9jdA/640?wx_fmt=png&from=appmsg&randomid=eepcet77&tp=wxpic&wxfrom=5&wx_lazy=1)

---

## 2. 核心框架

![分类框架](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGhKg9nnSz5qQrwKvXibt3wuhfgUpIfdPSqH8YjjHbCUiaaKsMA36bIMsMtGNKoBcus5py06M0fvx3A/640?wx_fmt=png&randomid=8uqao4g0&tp=wxpic&wxfrom=5&wx_lazy=1)

本综述的核心贡献是提出了一个将上下文工程分解为**基础组件（Foundational Components）**和**系统实现（System Implementations）**的分类框架。

### 2.1 上下文工程的定义与形式化

#### 2.1.1 基础定义

对于一个自回归的LLM，其模型参数为θ，在给定上下文C的条件下，生成输出序列的过程可以表示为最大化条件概率：

![条件概率公式](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDEOs7jjHFG4dlfw38NctrmQJDo5zJ7F7oCF4v4NlcrPngCsD4xBqQiaTgKYgW0W2v4VxoEAyrk8icicg/640?wx_fmt=png&from=appmsg&randomid=voojiflg&tp=wxpic&wxfrom=5&wx_lazy=1)

#### 2.1.2 从字符串到结构化集合

**传统提示工程**：
```
C = prompt  (单一文本字符串)
```

**上下文工程**：
```
C = {c₁, c₂, ..., cₙ}  (结构化信息组件集合)
```

这些组件由一系列函数进行获取、过滤和格式化，并最终由一个高阶的组装函数A进行编排：

![组装函数](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDEOs7jjHFG4dlfw38NctrmQPscJHB2HHqfqQRIw73TcVfaBb9TX7wp0ovgTVMdC8Xy6xNr2Wiaecgw/640?wx_fmt=png&from=appmsg&randomid=54p4z491&tp=wxpic&wxfrom=5&wx_lazy=1)

#### 2.1.3 核心上下文组件

这些组件对应了本综述的核心技术领域：

| 组件符号 | 含义 | 说明 |
|---------|------|------|
| c_inst | 指令组件 | 系统指令和规则 |
| c_know | 知识组件 | 通过RAG等功能检索到的外部知识 |
| c_tool | 工具组件 | 可用外部工具的定义和签名 |
| c_mem | 记忆组件 | 来自先前交互的持久化信息 |
| c_state | 状态组件 | 用户、世界或多智能体系统的动态状态 |
| c_query | 查询组件 | 用户的即时请求 |

#### 2.1.4 优化目标

上下文工程的优化问题可以定义为：寻找一组最优的上下文生成函数集合F（包括A、检索、选择等函数），以最大化LLM输出质量的期望值。

给定任务分布T，其目标是：

![优化目标公式](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDEOs7jjHFG4dlfw38NctrmQjEvvHe0bP7642aXHpWdDIWyvSjyBzdsLXjSOajzDrK08F6Oqe5odGA/640?wx_fmt=png&from=appmsg&randomid=76uxrvk1&tp=wxpic&wxfrom=5&wx_lazy=1)

其中：
- t 是一个具体的任务实例
- C(t) 是由函数集F为该任务生成的上下文
- y* 是理想的输出

**约束条件**：
- 模型上下文长度限制 L_max
- 计算资源约束
- 实时性要求

---

### 2.2 上下文工程的基础组件

![基础组件全景](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDEOs7jjHFG4dlfw38NctrmQPeQocyLStpLa7gdwExFtmMD1yQzzK3ZQI9RMqDxzCMtPjF64GwciaYw/640?wx_fmt=png&from=appmsg&randomid=jlr1qlho&tp=wxpic&wxfrom=5&wx_lazy=1)

基础组件是上下文工程的基石，我们将其分为**三个关键阶段**：

#### 2.2.1 上下文检索与生成（Context Retrieval and Generation）

该组件负责获取相关的上下文信息，包含三个主要机制：

**1. 提示工程与上下文生成**
- 设计有效的指令和推理框架
- 技术方法：
  - 思维链（Chain-of-Thought, CoT）
  - 思维树（Tree-of-Thoughts, ToT）
  - 思维图（Graph-of-Thoughts, GoT）
- 目标：引导模型的思考过程

**2. 外部知识检索**
- 通过检索增强生成（RAG）等技术
- 动态访问外部信息源：
  - 数据库
  - 知识图谱
  - 文档集合
- 目标：克服模型参数化知识的局限性

**3. 动态上下文组装**
- 将获取到的不同信息组件编排
- 组件类型：指令、知识、工具
- 目标：形成连贯的、为特定任务优化的上下文

#### 2.2.2 上下文处理（Context Processing）

该组件负责转换和优化获取到的信息，主要包括：

**1. 长上下文处理**
- **核心挑战**：Transformer架构的O(n²)复杂度问题
- **解决方案**：
  - 架构创新：状态空间模型（Mamba）
  - 位置插值技术
  - 优化技术：FlashAttention
- **目标**：处理超长序列

**2. 上下文自精炼与适应**
- **自精炼**：通过迭代反馈循环改进输出
  - 框架：Self-Refine
  - 机制：持续优化
- **快速适应**：
  - 元学习（Meta-Learning）
  - 记忆增强机制
  - 目标：适应新任务

**3. 多模态及结构化上下文**
- **多模态整合**：
  - 文本外数据：图像、音频
  - 统一表示空间
- **结构化数据**：
  - 知识图谱
  - 表格数据
- **当前挑战**：这是核心难题之一

#### 2.2.3 上下文管理（Context Management）

该组件关注上下文信息的有效组织、存储和利用：

**1. 基本约束**
- **上下文窗口限制**
- **"中间遗忘"现象**（lost-in-the-middle）
- **计算开销**的根本性限制

**2. 记忆层次与存储架构**
- **设计思想**：借鉴操作系统的虚拟内存管理
- **分层记忆系统**：
  - 典型实现：MemGPT
  - 机制：在有限上下文窗口和外部存储之间交换信息
- **架构层级**：
  - 工作记忆（上下文窗口内）
  - 外部存储（持久化）

**3. 上下文压缩**
- **目标**：减少计算和存储负担，同时保留关键信息
- **技术方法**：
  - 自编码器
  - 循环压缩
  - 基于记忆的方法
- **权衡**：压缩率 vs 信息保真度

---

### 2.3 上下文工程的系统实现

基础组件是构建更复杂的、面向应用的系统实现的基石。本综述探讨了**四种主要的系统实现方式**：

#### 2.3.1 检索增强生成（Retrieval-Augmented Generation, RAG）

![RAG系统架构](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDEOs7jjHFG4dlfw38NctrmQ7LibicAS1oibPd4icGPtnIhlSwE6NHlNrS08dCsdmrqAYsibkFARpEnOoPA/640?wx_fmt=png&from=appmsg&randomid=y7o7115m&tp=wxpic&wxfrom=5&wx_lazy=1)

RAG系统将外部知识源与LLM的生成过程相结合，已从简单的线性流程演变为更复杂的架构：

**1. 模块化RAG（Modular RAG）**
- **特点**：将RAG流程分解为可重新配置的模块
- **优势**：灵活的组件交互和定制化
- **应用**：企业级知识库系统

**2. 智能体RAG（Agentic RAG）**
- **特点**：将自主AI智能体嵌入RAG流程
- **能力**：
  - 持续的推理
  - 动态规划
  - 工具使用
- **优势**：动态管理检索策略

**3. 图增强RAG（Graph-Enhanced RAG）**
- **特点**：利用知识图谱等结构化知识表示
- **能力**：
  - 捕捉实体关系
  - 支持多跳推理
- **优势**：减少上下文漂移和幻觉

#### 2.3.2 记忆系统（Memory Systems）

![记忆系统架构](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDEOs7jjHFG4dlfw38NctrmQ3NKoTPkw1rJaMPMkiaYg2CJNARibWKak1HMnNAt31WWSyycwRPzDzdCA/640?wx_fmt=png&from=appmsg&randomid=kmbgg8bw&tp=wxpic&wxfrom=5&wx_lazy=1)

**核心价值**：使LLMs能够超越无状态的交互模式，实现信息的持久化存储、检索和利用。

**记忆层次**：
- **短期记忆**：在上下文窗口内操作
- **长期记忆**：利用外部数据库或专用结构

**应用领域**：
- 个性化对话
- 任务规划
- 社交模拟

**发展潜力**：记忆增强的智能体展现了巨大潜力。

#### 2.3.3 工具集成推理（Tool-Integrated Reasoning）

![工具集成架构](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDEOs7jjHFG4dlfw38NctrmQBIGqqnYBWF1xaHuV7YlWnOIXZt7gN4l6j2UqtrYiaImoRkuZETkAhkg/640?wx_fmt=png&from=appmsg&randomid=f8lvixu6&tp=wxpic&wxfrom=5&wx_lazy=1)

**核心转变**：将LLMs从被动的文本生成器转变为主动的世界交互者。

**实现机制**：
- **函数调用（Function Calling）**
- **外部工具**：
  - 计算器
  - 搜索引擎
  - API接口

**核心能力**：
- 克服内在的知识过时
- 提高计算准确性
- 扩展行动能力

**关键要求**：
- 自主选择合适的工具
- 解释中间输出
- 根据实时反馈调整策略

#### 2.3.4 多智能体系统（Multi-Agent Systems）

![多智能体系统](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDEOs7jjHFG4dlfw38NctrmQZY2x6ib5BRshgtkjnRK9SfwWDK0U0NE5JrenRQ3I7DIibxXGe16JFNzw/640?wx_fmt=png&from=appmsg&randomid=5bjsd4sg&tp=wxpic&wxfrom=5&wx_lazy=1)

**定位**：协作智能的顶峰

**核心机制**：
- 允许多个自主智能体协作
- 复杂的通信协议
- 编排机制
- 协调策略

**解决问题**：单个智能体无法完成的复杂任务

**LLMs的贡献**：极大地增强了智能体能力：
- 规划能力
- 专业化分工
- 任务分解

---

## 3. 小结与未来展望

![研究展望](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGhKg9nnSz5qQrwKvXibt3wukOjHSmSsEuRCB0fJu69CtdNgLnvFPDUCgeicOppBKuDvniaD3q8XWQ0Q/640?wx_fmt=png&randomid=q0blmy5s&tp=wxpic&wxfrom=5&wx_lazy=1)

### 3.1 核心贡献

本综述通过建立一个统一的分类框架，系统性地梳理了上下文工程这一新兴领域。

### 3.2 关键发现："理解-生成"差距

我们的分析揭示了一个关键的研究空白：

> **当前模型在理解复杂上下文方面表现出色，但在生成同等复杂的长篇输出方面存在明显局限。**

**弥合这一"理解-生成"差距是未来研究的重中之重。**

### 3.3 未来研究方向

#### 3.3.1 基础研究挑战

- **统一理论基础**：建立统一的理论基础和数学框架
- **缩放定律**：研究上下文工程的缩放定律
- **多模态整合**：解决多模态信息的整合与表示问题

#### 3.3.2 技术创新机遇

- **新架构探索**：状态空间模型等新一代架构
- **高级推理**：发展更高级的推理与规划能力
- **智能化组装**：实现智能化的上下文组装与优化

#### 3.3.3 应用驱动的研究

- **领域专业化**：针对特定领域（医疗、科研等）进行深度专业化
- **大规模协调**：实现大规模多智能体的协调
- **人机协同**：促进人与AI的协同工作

#### 3.3.4 部署和社会影响

- **可扩展性**：解决大规模部署的技术挑战
- **安全性与鲁棒性**：确保系统的安全性和可靠性
- **伦理问题**：处理AI系统的伦理影响
- **生产部署**：解决实际部署中的工程问题

---

## 4. 结语

上下文工程是推动AI系统从理论走向现实、从单一能力走向综合智能的关键。

我们希望这篇综述能为从事以下研究的读者提供系统性参考：
- 大模型系统
- Agent设计
- RAG架构
- 结构化数据融合

也欢迎关注我们的持续更新和后续研究进展。

---

## 参考资源

| 资源类型 | 链接 |
|---------|------|
| 论文全文 | https://arxiv.org/abs/2507.13334 |
| GitHub资源库 | https://github.com/Meirtz/Awesome-Context-Engineering |
| Hugging Face | https://huggingface.co/papers/2507.13334 |
| 中科院计算所 | http://www.ict.ac.cn/ |

---

**文档版本**：v1.0
**整理日期**：2025年12月
**原文来源**：中科院计算技术研究所
**整理说明**：本文档基于原始综述整理，保留了所有图片和核心内容，调整了格式以提高可读性。
