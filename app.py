# coding=utf-8
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from tester import OpenAITester

st.set_page_config(page_title="OpenAI API 测试工具箱", page_icon="🧪", layout="wide")

# 按钮骚粉色调样式
st.markdown("""
<style>
    /* 确保按钮可见性 */
    .stButton > button {
        background: linear-gradient(90deg, #ff69b4, #ff1493) !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.5rem 2rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        display: block !important;
        width: 100% !important;
        margin: 0.5rem 0 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff1493, #ff69b4) !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(255, 105, 180, 0.3) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    .stButton > button:focus {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(255, 105, 180, 0.5) !important;
    }
    .success-message {
        background: linear-gradient(90deg, #ff69b4, #ff1493);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 OpenAI API 测试工具箱")

# 初始化 session
if "tester" not in st.session_state:
    st.session_state.tester = None
if "history" not in st.session_state:
    st.session_state.history = []
if "test_results" not in st.session_state:
    st.session_state.test_results = []

# 侧边栏：直接输入配置
with st.sidebar:
    st.header("🔧 直接配置参数")
    base_url = st.text_input("Base URL", "https://api.openai.com/v1", help="例如: http://localhost:11434/v1")
    api_key = st.text_input("API Key", type="password", help="直接输入，不保存")
    model = st.text_input("Model Name", "gpt-3.5-turbo", help="例如: deepseek-r1, qwen2.5")
    timeout = st.number_input("Timeout (s)", 5, 300, 30)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.number_input("Max Tokens", 1, 16384, 4096)
    system_prompt = st.text_area("System Prompt", "You are a helpful assistant.", height=80)

    if st.button("🔄 初始化客户端"):
        if not base_url or not api_key or not model:
            st.error("Base URL、API Key、Model 不能为空！")
        else:
            try:
                st.session_state.tester = OpenAITester(base_url, api_key, model, timeout)
                st.success("✅ 客户端初始化成功！")
                st.session_state.history = []
            except Exception as e:
                st.error(f"初始化失败: {str(e)}")

# 主界面
tab1, tab2 = st.tabs(["💬 单轮测试", "🚀 并发压测"])

with tab1:
    if st.session_state.tester is None:
        st.warning("请先初始化客户端。")
    else:
        prompt = st.text_area("用户输入", height=100)
        col1, col2, col3 = st.columns([1, 1, 1])
        stream_mode = col1.checkbox("流式输出", True)
        send_btn = col2.button("发送")
        clear_btn = col3.button("清空历史")

        if clear_btn:
            st.session_state.history = []

        if send_btn and prompt:
            with st.spinner("正在请求..."):
                res = st.session_state.tester.single_chat(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream_mode
                )
            st.session_state.history.append({"prompt": prompt, **res})

        for h in reversed(st.session_state.history):
            with st.chat_message("user"):
                st.write(h["prompt"])
            with st.chat_message("assistant"):
                if h["success"]:
                    if h["reasoning"]:
                        with st.expander("🔍 推理过程"):
                            st.markdown(h["reasoning"])
                    st.markdown(h["response"])
                    st.caption(f"耗时: {h['time']}s")
                else:
                    st.error(f"错误: {h['error']}")

with tab2:
    if st.session_state.tester is None:
        st.warning("请先初始化客户端。")
    else:
        st.subheader("并发压力测试")
        test_prompt = st.text_input("测试问题", "你好，请告诉我今天的天气。")
        
        # 测试模式选择
        test_mode = st.radio("测试模式", ["固定时长", "固定请求数"], horizontal=True)
        
        if test_mode == "固定请求数":
            total = st.number_input("总请求数", 1, 1000, 50)
            concur = st.number_input("并发数", 1, 100, 10)
        else:  # 固定时长
            duration = st.number_input("测试时长（秒）", 10, 3600, 60, help="持续测试指定时长")
            concur = st.number_input("并发数", 1, 100, 10)
        
        run_btn = st.button("🚀 开始测试")

        if run_btn and test_prompt:
            if test_mode == "固定请求数":
                with st.spinner("测试中..."):
                    stats = st.session_state.tester.concurrent_test(
                        prompt=test_prompt,
                        total=total,
                        concurrency=concur,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                st.markdown('<div class="success-message">✅ 测试完成！</div>', unsafe_allow_html=True)
            else:  # 固定时长
                # 创建状态显示容器
                status_container = st.empty()
                live_text_container = st.empty()
                
                # 显示测试开始信息
                status_container.info(f"🚀 开始固定时长测试: {duration}秒 / {concur} 并发")
                
                try:
                    # 使用多线程来实时更新界面显示，同时避免闪烁，展示文本统计
                    import threading
                    import time
                    
                    # 从主线程捕获 tester 引用
                    tester_ref = st.session_state.tester
                    if tester_ref is None:
                        status_container.error("请先在左侧初始化客户端后再开始测试")
                        stats = None
                        raise RuntimeError("Tester not initialized")
                    
                    # 存储测试结果与实时进度
                    test_result = {"stats": None, "error": None}
                    latest = {"elapsed": 0.0, "target": duration, "requests": 0, "success": 0, "qps": 0.0}
                    latest_lock = __import__('threading').Lock()
                    
                    def progress_cb(data: dict):
                        with latest_lock:
                            latest.update(data)
                    
                    def run_test():
                        try:
                            result = tester_ref.duration_test(
                                prompt=test_prompt,
                                duration=duration,
                                concurrency=concur,
                                system_prompt=system_prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                show_progress=False,  # 关闭终端进度条，使用UI文本
                                progress_callback=progress_cb
                            )
                            test_result["stats"] = result
                        except Exception as e:
                            test_result["error"] = str(e)
                    
                    # 启动测试线程
                    test_thread = threading.Thread(target=run_test)
                    test_thread.start()
                    
                    # 实时更新界面文本（不使用进度条）
                    while test_thread.is_alive():
                        with latest_lock:
                            elapsed = latest.get("elapsed", 0.0)
                            target = latest.get("target", duration)
                            requests = latest.get("requests", 0)
                            success = latest.get("success", 0)
                            qps = latest.get("qps", 0.0)
                        # 纠正显示范围，避免 2/60 或 62/60 误差
                        show_elapsed = min(max(elapsed, 0.0), float(target))
                        live_text_container.info(f"⏱️ {show_elapsed:.2f}s/{int(target)}s, requests={requests}, success={success}, qps={qps:.2f}")
                        time.sleep(1)
                    
                    # 等待测试完成
                    test_thread.join()
                    
                    if test_result["error"]:
                        status_container.error(f"测试执行失败: {test_result['error']}")
                        live_text_container.empty()
                        stats = None
                    else:
                        stats = test_result["stats"]
                        status_container.success("✅ 固定时长测试完成！")
                        # 最终再展示一次摘要
                        with latest_lock:
                            elapsed = latest.get("elapsed", stats.get('duration', 0.0))
                            target = latest.get("target", stats.get('target_duration', duration))
                            requests = stats.get('total', 0)
                            success = stats.get('success', 0)
                            qps = stats.get('qps', 0.0)
                        live_text_container.success(f"⏱️ {min(elapsed, target):.2f}s/{int(target)}s, requests={requests}, success={success}, qps={qps:.2f}")
                        st.markdown('<div class="success-message">✅ 固定时长测试完成！</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    status_container.error(f"测试执行失败: {str(e)}")
                    live_text_container.empty()
                    stats = None
            
            # 只有在测试成功时才存储和显示结果
            if stats is not None:
                # 存储测试结果
                test_result = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "test_mode": test_mode,
                    "prompt": test_prompt,
                    "concurrency": concur,
                    "stats": stats
                }
                if test_mode == "固定请求数":
                    test_result["total_requests"] = total
                else:
                    test_result["duration"] = duration
                
                st.session_state.test_results.append(test_result)
                
                # 显示测试结果
                if test_mode == "固定请求数":
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("总请求", stats["total"])
                    c2.metric("成功", stats["success"], f"{stats['success_rate']}%")
                    c3.metric("失败", stats["failed"])
                    c4.metric("QPS", stats["qps"])
                    c1, c2 = st.columns(2)
                    c1.metric("平均耗时", f"{stats['avg_time']}s")
                    c2.metric("P95 耗时", f"{stats['p95_time']}s")
                    c1, c2 = st.columns(2)
                    c1.metric("总耗时", f"{stats['total_wall_time']}s")
                else:  # 固定时长
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("测试时长", f"{stats['duration']}s", f"目标: {stats['target_duration']}s")
                    c2.metric("总请求", stats["total"])
                    c3.metric("成功", stats["success"], f"{stats['success_rate']}%")
                    c4.metric("失败", stats["failed"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("QPS", stats["qps"])
                    c2.metric("平均耗时", f"{stats['avg_time']}s")
                    c3.metric("P95 耗时", f"{stats['p95_time']}s")

                if stats.get("failures"):
                    with st.expander("⚠️ 失败请求"):
                        for e in stats["failures"]:
                            st.error(e)
            else:
                st.error("测试失败，无法显示结果")

# 导出功能和历史记录（移到测试区域外）
st.markdown("---")
st.subheader("📊 导出测试报告")

col1, col2, col3 = st.columns(3)

with col1:
    # 导出为Excel
    if st.button("📈 导出Excel报告"):
        if st.session_state.test_results:
            df_data = []
            for result in st.session_state.test_results:
                if result.get("stats"):  # 确保stats存在
                    row = {
                        "测试时间": result["timestamp"],
                        "测试模式": result["test_mode"],
                        "并发数": result["concurrency"],
                        "总请求数": result["stats"]["total"],
                        "成功数": result["stats"]["success"],
                        "失败数": result["stats"]["failed"],
                        "成功率(%)": result["stats"]["success_rate"],
                        "平均耗时(s)": result["stats"]["avg_time"],
                        "P95耗时(s)": result["stats"]["p95_time"],
                        "QPS": result["stats"]["qps"]
                    }
                    if result["test_mode"] == "固定请求数":
                        row["总请求数"] = result.get("total_requests", 0)
                        row["总耗时(s)"] = result["stats"].get("total_wall_time", 0)
                    else:
                        row["测试时长(s)"] = result.get("duration", 0)
                        row["实际时长(s)"] = result["stats"].get("duration", 0)
                    df_data.append(row)
            
            if df_data:
                df = pd.DataFrame(df_data)
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="💾 下载Excel报告",
                    data=csv,
                    file_name=f"api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.success("✅ Excel报告已准备就绪，点击下载按钮保存文件")
            else:
                st.warning("暂无测试数据可导出")
        else:
            st.warning("暂无测试数据可导出")

with col2:
    # 导出为JSON
    if st.button("📄 导出JSON报告"):
        if st.session_state.test_results:
            json_data = json.dumps(st.session_state.test_results, ensure_ascii=False, indent=2)
            st.download_button(
                label="💾 下载JSON报告",
                data=json_data,
                file_name=f"api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            st.success("✅ JSON报告已准备就绪，点击下载按钮保存文件")
        else:
            st.warning("暂无测试数据可导出")

with col3:
    # 清空历史记录
    if st.button("🗑️ 清空测试记录"):
        st.session_state.test_results = []
        st.success("✅ 测试记录已清空")
        st.rerun()

# 显示历史测试记录
if st.session_state.test_results:
    st.subheader("📋 历史测试记录")
    for i, result in enumerate(reversed(st.session_state.test_results[-5:])):  # 只显示最近5条
        if result.get("stats"):  # 确保stats存在
            with st.expander(f"测试记录 {i+1} - {result['timestamp']} ({result['test_mode']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**测试模式**: {result['test_mode']}")
                    st.write(f"**并发数**: {result['concurrency']}")
                    st.write(f"**总请求数**: {result['stats']['total']}")
                    st.write(f"**成功率**: {result['stats']['success_rate']}%")
                with col2:
                    st.write(f"**平均耗时**: {result['stats']['avg_time']}s")
                    st.write(f"**P95耗时**: {result['stats']['p95_time']}s")
                    st.write(f"**QPS**: {result['stats']['qps']}")
                    if result['test_mode'] == "固定请求数":
                        st.write(f"**总耗时**: {result['stats'].get('total_wall_time', 0)}s")
                    else:
                        st.write(f"**测试时长**: {result.get('duration', 0)}s")
