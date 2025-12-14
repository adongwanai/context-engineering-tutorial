# 代码示例目录

本目录包含教程中所有实战代码示例，支持多种主流Agent框架。

---

## 📁 目录结构

```
examples/
├── langraph/              # LangGraph示例（主框架）
├── dify/                  # Dify配置文件
├── autogen/              # AutoGen示例
├── crewai/               # CrewAI示例
└── framework-comparison/ # 框架对比代码
```

---

## 🚀 快速开始

### LangGraph示例

```bash
cd langraph
pip install -r requirements.txt
python example_rag.py
```

### Dify配置

Dify使用可视化配置，请导入 `dify/` 目录下的配置文件。

### AutoGen示例

```bash
cd autogen
pip install -r requirements.txt
python example_multi_agent.py
```

---

## 📊 框架对比

| 框架 | 适合场景 | 示例文件 |
|------|---------|---------|
| LangGraph | 复杂流程、状态管理 | `langraph/example_rag.py` |
| Dify | 快速原型、可视化 | `dify/智能客服.json` |
| AutoGen | 对话式多Agent | `autogen/example_multi_agent.py` |
| CrewAI | 团队协作 | `crewai/example_crew.py` |

---

## 🔗 相关章节

- 第9章：Agent架构设计 - 框架对比
- 第10章：工具集成 - 工具调用示例
- 第17章：完整项目实战 - 综合应用

---

**注意**：所有示例需要配置相应的API密钥，请参考各框架目录下的 `.env.example` 文件。






