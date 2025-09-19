# OpenAI API 测试工具箱

一个支持 **可视化界面** 和 **命令行快速测试** 的 OpenAI 协议 API 测试工具，适合做固定时长的持续负载压测与并发能力评估。

## 📦 安装

```bash
pip install -r requirements.txt
```

可选：创建虚拟环境

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

---

## 🖥️ 方式一：使用 Streamlit 可视化界面（推荐）

适合：调试、查看推理内容、团队共享。

### 启动命令

```bash
streamlit run app.py
```

### 配置说明（直接在界面填写）：

| 参数 | 示例 | 说明 |
|------|------|------|
| Base URL | `https://api.deepseek.com/v1` | API 地址 |
| API Key | `sk-xxxxxx` | 直接输入，不保存 |
| Model Name | `deepseek-r1` | 模型名 |
| Timeout | `30` | 超时秒数 |
| 其他参数 | — | 可调整温度、max_tokens 等 |

> 💡 支持 **流式输出**、**推理内容展示**、**固定时长并发压测**、**报告导出**

---

## ⚡ 方式二：使用命令行快速测试（无需界面）

适合：自动化、CI/CD、快速验证连通性和最大并发能力。

### 示例 1：测试连通性

```bash
python cli_tester.py --base-url "https://api.deepseek.com/v1" --api-key "sk-xxxxxx" --model "deepseek-r1" --prompt "你好"
```

### 示例 2：压测最大并发能力（找出瓶颈）

```bash
# 逐步增加并发，找到成功率下降点
python cli_tester.py --base-url "http://localhost:11434/v1" --api-key "ollama" --model "qwen2.5" --total 100 --concurrency 20 --prompt "写一首诗"
```

### 示例 3：固定时长持续负载测试（推荐评估）

```bash
# 持续测试60秒，观察系统在持续负载下的表现
python cli_tester.py --base-url "https://api.deepseek.com/v1" --api-key "sk-xxxxxx" --model "deepseek-r1" --duration 60 --concurrency 10 --prompt "请分析一下人工智能的发展趋势"
```

### 常见问题（FAQ）

- 终端看不到进度条？
  - 使用 CLI 时会显示 tqdm 进度条；Web 界面内我们以文本方式更新实时信息，避免闪烁。
- 导出报告后页面内容消失？
  - 已修复：导出区域已移至结果展示区域之外，下载后不再清空内容。
- 固定时长测试没有尽头感？
  - Web 界面实时显示 `elapsed/target, requests, success, qps` 文本，避免焦虑感。

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--base-url` | ✅ | — | API 地址 |
| `--api-key` | ✅ | — | API Key |
| `--model` | ✅ | — | 模型名 |
| `--timeout` | ❌ | 30 | 超时秒数 |
| `--prompt` | ❌ | "你好..." | 测试问题 |
| `--total` | ❌ | 10 | 总请求数（与 --duration 互斥） |
| `--duration` | ❌ | — | 固定时长测试（秒，与 --total 互斥） |
| `--concurrency` | ❌ | 5 | 并发数（建议从 5 开始逐步增加） |
| `--temperature` | ❌ | 0.7 | 温度 |
| `--max-tokens` | ❌ | 4096 | 最大输出 token |

### 并发压测推荐流程（固定时长优先）

1) 基线测试（单线程性能）

```bash
python cli_tester.py --base-url xxx --api-key sk-xxx --model xxx \
  --concurrency 1 --duration 60 --prompt "hello"
```

2) 阶梯增压测试（持续负载）

```bash
# 建议阶梯：10 -> 20 -> 35 -> 50，并保持 60~180 秒
python cli_tester.py --base-url xxx --api-key sk-xxx --model xxx --concurrency 10 --duration 120
python cli_tester.py --base-url xxx --api-key sk-xxx --model xxx --concurrency 20 --duration 120
# ... 继续提升
```

3) 评估与结论（三个核心指标）

- 成功率：是否 100%（不为 100% 则失败）
- 延迟：P95 是否在可接受范围（如 ≤ 15s）
- 吞吐(QPS)：是否随并发提高接近线性增长

当 `--concurrency 50 --duration 180` 满足：

- 成功率 == 100%
- P95 < 目标值（如 15s）
- QPS 相比基线明显提升

即可认为该模型在此配置下可稳定支持 50 路并发。

### 测试模式对比

| 模式 | 参数 | 适用场景 | 特点 |
|------|------|----------|------|
| **固定请求数** | `--total` | 快速验证、找出并发上限 | 瞬时压力，容易“尖峰” |
| **固定时长** | `--duration` | 持续负载、稳定性验证 | 持续压力，更接近真实场景（推荐） |

---

## 🔐 安全提示

- API Key 仅在内存中使用，**不会写入文件**

## 🧩 支持模型

- GPT 系列
- DeepSeek-R1/V3
- Qwen
- Ollama
- 所有兼容 OpenAI 协议的模型

## 🚀 适用场景

- 模型连通性验证
- 并发能力压测（固定请求数 / 固定时长）
- 持续负载测试（固定时长，推荐）
- 推理模型调试
- 团队 API 测试标准工具
