# coding=utf-8
import time
import threading
from openai import OpenAI
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class OpenAITester:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 30):
        self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=timeout)
        self.model = model

    def single_chat(self, prompt: str, system_prompt: str = "You are a helpful assistant.",
                    temperature: float = 0.7, max_tokens: int = 4096, stream: bool = False) -> Dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            if stream:
                full_response = ""
                reasoning_content = ""
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        full_response += delta.content
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        reasoning_content += delta.reasoning_content
                return {
                    "success": True,
                    "response": full_response,
                    "reasoning": reasoning_content,
                    "time": round(time.time() - start_time, 3),
                    "error": None
                }
            else:
                msg = response.choices[0].message
                reasoning = getattr(msg, "reasoning_content", "")
                content = msg.content or ""
                return {
                    "success": True,
                    "response": content,
                    "reasoning": reasoning,
                    "time": round(time.time() - start_time, 3),
                    "error": None
                }
        except Exception as e:
            return {
                "success": False,
                "response": "",
                "reasoning": "",
                "time": round(time.time() - start_time, 3),
                "error": str(e)
            }

    def concurrent_test(self, prompt: str, total: int, concurrency: int, system_prompt: str = "",
                        temperature: float = 0.7, max_tokens: int = 4096, show_progress: bool = True) -> Dict[str, Any]:
        results = []
        start_wall_time = time.time()  # ✅ 记录开始时间

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self.single_chat, prompt, system_prompt, temperature, max_tokens, False)
                for _ in range(total)
            ]
            # 进度条
            iterator = tqdm(as_completed(futures), total=total, desc="并发测试", disable=not show_progress)
            for future in iterator:
                results.append(future.result())

        end_wall_time = time.time()  # ✅ 记录结束时间
        total_wall_time = end_wall_time - start_wall_time  # ✅ 墙钟时间

        # 统计
        success_list = [r for r in results if r["success"]]
        times = [r["time"] for r in success_list]
        p95 = sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else 0
        avg_time = sum(times) / len(times) if times else 0

        # ✅ 正确 QPS 计算
        qps = len(success_list) / total_wall_time if total_wall_time > 0 else 0

        # 失败信息
        failures = [f"Req-{r['time']}s: {r['error']}" for r in results if not r['success']][:5]

        return {
            "total": total,
            "success": len(success_list),
            "failed": total - len(success_list),
            "success_rate": round(len(success_list) / total * 100, 2),
            "avg_time": round(avg_time, 3),
            "p95_time": round(p95, 3),
            "qps": round(qps, 2),
            "total_wall_time": round(total_wall_time, 3),  # ✅ 新增：总测试时间
            "failures": failures
        }

    def duration_test(self, prompt: str, duration: int, concurrency: int, system_prompt: str = "",
                      temperature: float = 0.7, max_tokens: int = 4096, show_progress: bool = True,
                      progress_callback: Any = None) -> Dict[str, Any]:
        """
        固定时长测试模式：在指定时间内持续发送请求
        
        Args:
            prompt: 测试问题
            duration: 测试时长（秒）
            concurrency: 并发数
            system_prompt: 系统提示词
            temperature: 温度
            max_tokens: 最大tokens
            show_progress: 是否显示进度条
            
        Returns:
            测试统计结果
        """
        results = []
        start_time = time.time()
        end_time = start_time + duration
        stop_flag = threading.Event()
        
        def worker():
            """工作线程：持续发送请求直到时间结束"""
            while not stop_flag.is_set() and time.time() < end_time:
                result = self.single_chat(prompt, system_prompt, temperature, max_tokens, False)
                results.append(result)
                # 如果当前时间已经超过结束时间，立即停止
                if time.time() >= end_time:
                    break
        
        print(f"🚀 开始固定时长测试: {duration}秒 / {concurrency} 并发")
        print(f"⏰ 测试将在 {time.strftime('%H:%M:%S', time.localtime(end_time))} 结束")
        
        # 启动工作线程
        threads = []
        for _ in range(concurrency):
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # 实时进度反馈或进度条
        if show_progress and progress_callback is None:
            with tqdm(total=duration, desc="持续测试", unit="s") as pbar:
                while time.time() < end_time:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        break
                    pbar.update(1)
                    time.sleep(1)
                    success_cnt = len([r for r in results if r['success']])
                    elapsed = max(1e-6, time.time() - start_time)
                    pbar.set_postfix({
                        'requests': len(results),
                        'success': success_cnt,
                        'qps': round(success_cnt / elapsed, 2)
                    })
        else:
            # 使用回调提供实时进度（每秒一次），或简单等待
            while time.time() < end_time:
                if progress_callback is not None:
                    success_cnt = len([r for r in results if r['success']])
                    elapsed = max(1e-6, time.time() - start_time)
                    progress_callback({
                        'elapsed': round(elapsed, 2),
                        'target': duration,
                        'requests': len(results),
                        'success': success_cnt,
                        'qps': round(success_cnt / elapsed, 2)
                    })
                time.sleep(1)
        
        # 停止所有线程
        stop_flag.set()
        for thread in threads:
            thread.join(timeout=1)  # 最多等待1秒
        
        actual_duration = time.time() - start_time
        
        # 统计结果
        success_list = [r for r in results if r["success"]]
        times = [r["time"] for r in success_list]
        p95 = sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else 0
        avg_time = sum(times) / len(times) if times else 0
        
        # 计算QPS（基于实际测试时长）
        qps = len(success_list) / actual_duration if actual_duration > 0 else 0
        
        # 失败信息
        failures = [f"Req-{r['time']}s: {r['error']}" for r in results if not r['success']][:5]
        
        return {
            "total": len(results),
            "success": len(success_list),
            "failed": len(results) - len(success_list),
            "success_rate": round(len(success_list) / len(results) * 100, 2) if results else 0,
            "avg_time": round(avg_time, 3),
            "p95_time": round(p95, 3),
            "qps": round(qps, 2),
            "duration": round(actual_duration, 3),
            "target_duration": duration,
            "failures": failures
        }

