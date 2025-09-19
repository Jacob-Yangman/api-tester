# coding=utf-8


# python cli_tester.py --base-url https://xiaoai.plus/v1 --api-key sk-xxx --model deepseek-r1

'''
# 如何测试模型并发上限

concurrency为并发数，duration为压测的持续时长，至少设为120秒

指标解读：
- "响应时间: 0.0001s"表示冷启动时单次响应时间
- 平均耗时： 处理每个请求所花费的平均时间（从发送到收到完整响应），这个时间比冷启动时单次连通测试更快
- P95耗时 (P95 Latency)： 95%的请求所花费的时间
    例如"P95 耗时：6.086s"它表示95%的请求都在6.086秒以内完成了。只有5%的“慢请求”耗时比这个长
    这个指标说明了系统体验的“平滑度”。用户大部分时候感觉很快（平均耗时），但偶尔可能会遇到一些接近6秒的等待。这个数值是评估用户体验的关键
- QPS (Queries Per Second): 10.19，表示模型每秒可处理10个左右请求



第1步：基线测试（单线程性能）
python cli_tester.py --base-url xxx --api-key sk-xxx --model xxx --concurrency 1 --duration 60

第2步：阶梯增压测试
按照 10 -> 20 -> 35 -> 50 的并发数阶梯进行测试，每次测试时长至少60-120秒。
python cli_tester.py --base-url xxx --api-key sk-xxx --model xxx --concurrency 10 --duration 120
python cli_tester.py --base-url xxx --api-key sk-xxx --model xxx --concurrency 20 --duration 120
...


第3步：分析数据，得出结论
对每一次测试，观察三个核心指标：

1. 成功率：是否100%？不是则失败。
2. 延迟：P95延迟是否在业务可接受范围内？（例如 ≤15s）
3. 吞吐量 (QPS)：是否随着并发数增长而接近线性增长（比如从concurrency从10到20，QPS从1增长到2）

如果 --concurrency 50 --duration 180 的测试结果满足：
成功率 == 100%
P95延迟 < [你的目标值，如15s]
QPS 相比基线有显著提升
那么你就可以得出结论：该模型在此配置下可稳定支持50路并发。

'''

import argparse
from tester import OpenAITester

def main():
    parser = argparse.ArgumentParser(description="OpenAI API 快速连通性 & 并发测试")
    parser.add_argument("--base-url", required=True, help="API 地址，如 https://api.openai.com/v1")
    parser.add_argument("--api-key", required=True, help="API Key")
    parser.add_argument("--model", required=True, help="模型名，如 gpt-3.5-turbo")
    parser.add_argument("--timeout", type=int, default=30, help="超时时间（秒）")
    parser.add_argument("--prompt", default="你好，ChatGPT", help="测试问题")
    parser.add_argument("--total", type=int, default=10, help="并发总请求数")
    parser.add_argument("--concurrency", type=int, default=5, help="并发数")
    parser.add_argument("--duration", type=int, help="固定时长测试模式（秒），与--total互斥")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--max-tokens", type=int, default=4096, help="最大 tokens")

    args = parser.parse_args()

    print("🚀 正在初始化客户端...")
    tester = OpenAITester(args.base_url, args.api_key, args.model, args.timeout)

    # 步骤1：连通性测试
    print("🔍 正在进行连通性测试...")
    res = tester.single_chat(args.prompt, temperature=args.temperature, max_tokens=args.max_tokens)
    if res["success"]:
        print(f"✅ 连通成功！响应时间: {res['time']}s")
        if res["reasoning"]:
            print(f"🔍 推理内容: {res['reasoning']}")
        print(f"📝 回答: {res['response'][:100]}...")
    else:
        print(f"❌ 连通失败: {res['error']}")
        return

    # 步骤2：并发测试
    if args.duration:
        # 固定时长测试模式
        if args.total != 10:  # 如果用户同时指定了total和duration
            print("⚠️  警告: 固定时长模式下忽略--total参数")
        print(f"\n🚀 开始固定时长测试: {args.duration}秒 / {args.concurrency} 并发")
        stats = tester.duration_test(
            prompt=args.prompt,
            duration=args.duration,
            concurrency=args.concurrency,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        print("\n📊 固定时长测试结果:")
        print(f"  测试时长: {stats['duration']}s (目标: {stats['target_duration']}s)")
        print(f"  总请求: {stats['total']}")
        print(f"  成功: {stats['success']} ({stats['success_rate']}%)")
        print(f"  失败: {stats['failed']}")
        print(f"  平均耗时: {stats['avg_time']}s")
        print(f"  P95 耗时: {stats['p95_time']}s")
        print(f"  QPS: {stats['qps']}")
    else:
        # 固定请求数测试模式
        print(f"\n🚀 开始并发测试: {args.total} 请求 / {args.concurrency} 并发")
        stats = tester.concurrent_test(
            prompt=args.prompt,
            total=args.total,
            concurrency=args.concurrency,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        print("\n📊 测试结果:")
        print(f"  总请求: {stats['total']}")
        print(f"  成功: {stats['success']} ({stats['success_rate']}%)")
        print(f"  失败: {stats['failed']}")
        print(f"  平均耗时: {stats['avg_time']}s")
        print(f"  P95 耗时: {stats['p95_time']}s")
        print(f"  QPS: {stats['qps']}")
        print(f"  总耗时: {stats['total_wall_time']}s")
    
    if stats["failures"]:
        print("  部分错误:")
        for e in stats["failures"]:
            print(f"    - {e}")

if __name__ == "__main__":
    main()
