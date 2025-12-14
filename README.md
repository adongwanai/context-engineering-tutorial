# 《上下文工程实战：让AI真正懂你的需求》

> **教程定位**：系统化的上下文工程学习路径，从理论认知到生产实践的完整知识体系
>
  **核心理念**：上下文工程是将不断变化的信息宇宙中最相关内容，精心筛选并放入有限上下文窗口的艺术与科学——本质是"熵减"过程

## 📖 序章

在开始学习之前，强烈建议先阅读 **[序章：上下文工程完全指南](context-engineering-guide.md)**，它将帮助你：
- 理解什么是上下文工程及其核心挑战
- 了解上下文工程的6大核心组件
- 建立完整的上下文工程知识框架
- 掌握从简单提示到复杂系统的演进路径

序章涵盖：Agents(决策大脑)、Query Augmentation(查询增强)、Retrieval(检索系统)、Prompting Techniques(提示技巧)、Memory(记忆系统)、Tools(工具集成)

---

## 📚 教程结构（17章）

### 第一部分：从提示词到上下文 - 建立正确认知（第1-3章）
- [第1章：什么是上下文工程？](docs/part1-intro/chapter01-what-is-context-engineering.md)
- [第2章：核心原理 - 为什么Context有效？](docs/part1-intro/chapter02-why-context-works.md)
- [第3章：上下文失败模式与诊断](docs/part1-intro/chapter03-context-failure-patterns.md)

### 第二部分：核心技术 - 提示、查询、意图识别（第4-6章）
- [第4章：提示技巧（Prompting Techniques）](docs/part2-fundamentals/chapter04-prompting-techniques.md)
- [第5章：查询增强（Query Augmentation）](docs/part2-fundamentals/chapter05-query-augmentation.md)
- [第6章：意图识别与路由](docs/part2-fundamentals/chapter06-intent-recognition.md)

### 第三部分：实用技巧 - 检索和记忆怎么做（第7-8章）
- [第7章：检索系统（Retrieval/RAG）](docs/part3-practical/chapter07-retrieval-rag.md)
- [第8章：记忆系统（Memory）](docs/part3-practical/chapter08-memory-systems.md)

### 第四部分：搭建Agent - 从架构到优化（第9-12章）
- [第9章：Agent架构设计](docs/part4-development/chapter09-agent-architecture.md)
- [第10章：工具集成（Tools）](docs/part4-development/chapter10-tools-integration.md)
- [第11章：12-Factor Agents工程化原则](docs/part4-development/chapter11-12-factor-agents.md)
- [第12章：强化学习驱动的Agent优化](docs/part4-development/chapter12-agentic-rl.md)

### 第五部分：高级技巧 - 自动化、评估、安全（第13-15章）
- [第13章：自动化与编译式上下文](docs/part5-advanced/chapter13-dspy.md)
- [第14章：评估与监控](docs/part5-advanced/chapter14-evaluation-monitoring.md)
- [第15章：安全与攻防](docs/part5-advanced/chapter15-security.md)

### 第六部分：实战落地 - 看案例学实战（第16-17章）
- [第16章：企业级实战案例](docs/part6-production/chapter16-enterprise-cases.md)
- [第17章：完整项目实战](docs/part6-production/chapter17-complete-projects.md)

---

## 🎯 学习路径

### 入门路径（1-2周）
1. 阅读第1-3章：建立上下文工程的基本认知
2. 实践第4章：掌握基础提示技巧
3. 了解常见失败模式，避坑

### 进阶路径（2-4周）
4. 学习第5-8章：掌握查询、意图、RAG、记忆核心技术
5. 阅读第9章：了解主流Agent框架对比
6. 选择一个框架开始实践

### 专家路径（1-3个月）
7. 深入第10-12章：工具集成、工程化、RL优化
8. 学习第13-15章：自动化优化、评估监控、安全
9. 实战第16-17章：企业案例分析、完整项目开发

---

## 💻 代码示例

本教程提供多框架代码示例：

- **LangGraph示例**：`examples/langraph/`
- **Dify配置**：`examples/dify/`
- **AutoGen示例**：`examples/autogen/`
- **CrewAI示例**：`examples/crewai/`
- **框架对比**：`examples/framework-comparison/`

---

## 📖 资源索引

- **学术论文**：21篇核心论文
- **工具框架**：10个主流框架
- **评估工具**：6个专业评估平台
- **企业案例**：Manus、Anthropic等实战经验
- **总计**：63项精选资源

详见：[完整资源列表](context-engineering-tutorial-outilne.md)

---

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/adongwanai/context-engineering-tutorial.git

# 进入目录
cd context-engineering-tutorial

# 阅读大纲
cat context-engineering-tutorial-outilne.md

# 开始学习第1章
cat docs/part1-intro/chapter01-what-is-context-engineering.md
```

---

## 🤝 贡献指南

欢迎贡献！请查看 [贡献指南](CONTRIBUTING.md)

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📮 联系方式

- 问题反馈：[GitHub Issues](https://github.com/adongwanai/context-engineering-tutorial/issues)
- 讨论交流：[GitHub Discussions](https://github.com/adongwanai/context-engineering-tutorial/discussions)

---

**版本**：v1.0
**最后更新**：2025年12月
**作者**：adongwanai






