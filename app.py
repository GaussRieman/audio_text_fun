import streamlit as st
import os
import time
import pandas as pd
from model import ParaformerModel, SenseVoiceModel
from qwen_llm import (
    get_qa_pairs_from_text_stream,
    extract_qa_pairs_from_llm_result,
    QA_EXTRACTION_PROMPT
)
import re


# 页面配置
st.set_page_config(
    page_title="音频文本处理系统",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tab-content {
        padding: 2rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 全局变量存储模型
asr_model = None
model_loaded = False

@st.cache_resource
def get_asr_model():
    global asr_model, model_loaded
    if asr_model is None:
        try:
            asr_model = ParaformerModel()
            model_loaded = True
            return asr_model, asr_model.device, None
        except Exception as e:
            return None, None, str(e)
    return asr_model, asr_model.device, None

def clear_results():
    if 'transcribed_text' in st.session_state:
        del st.session_state.transcribed_text
    if 'asr_stats' in st.session_state:
        del st.session_state.asr_stats
    st.rerun()

def process_audio(uploaded_file, hotword=None):
    """处理音频文件"""
    try:
        model, device, err = get_asr_model()
        if err:
            st.error(f"❌ ASR模型加载失败: {err}")
            return
        if not model:
            st.error("ASR模型未加载")
            return
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("🎵 正在转写音频...")
        progress_bar.progress(25)
        start_time = time.time()
        
        # 处理热词参数 - 保持空格分隔格式
        hotword_str = hotword.strip() if hotword else None
        if hotword_str:
            st.info(f"🔥 使用热词: {hotword_str}")
            print(f"热词参数: {hotword_str}")
        
        transcribed_text = model.transcribe(audio_path, hotword=hotword_str)
        progress_bar.progress(100)
        status_text.text("✅ 转写完成!")
        processing_time = time.time() - start_time
        st.session_state.transcribed_text = transcribed_text
        st.session_state.asr_stats = {
            'time': processing_time,
            'text_length': len(transcribed_text),
            'file_name': uploaded_file.name,
            'hotword_used': hotword_str
        }
        st.success(f"🎉 转写完成！耗时 {processing_time:.2f} 秒")
        if uploaded_file and os.path.exists(audio_path):
            os.remove(audio_path)
        try:
            st.rerun()
        except Exception:
            pass
    except Exception as e:
        st.error(f"❌ 转写失败: {str(e)}")
        st.exception(e)

def asr_tab():
    """ASR音频转写功能"""
    st.markdown('<h2 class="main-header">🎤 音频转写 (ASR)</h2>', unsafe_allow_html=True)
    model, device, err = get_asr_model()
    if err:
        st.error(f"❌ ASR模型加载失败: {err}")
        return
    if not model:
        st.info("🔄 正在初始化ASR模型，请稍候...")
        return
    if not st.session_state.get("asr_model_loaded_toast", False):
        st.session_state["asr_model_loaded_toast"] = True
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ ASR配置")
        
        # 显示模型信息
        st.markdown("### 📋 模型信息")
        st.info(f"设备: {'GPU' if 'cuda' in device else 'CPU'}")
        st.info("模型: SenseVoiceSmall")
        
        st.markdown("### 📊 ASR处理统计")
        # 统计每次都从 session_state 读取，保证自动刷新
        stats = st.session_state.get('asr_stats', {})
        st.metric("处理时间", f"{stats.get('time', 0):.2f}秒")
        st.metric("文本长度", f"{stats.get('text_length', 0)}字符")
        if stats.get('hotword_used'):
            st.info(f"🔥 热词: {stats.get('hotword_used')}")
    
    # 主要内容区域
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 音频文件上传")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择音频文件",
            type=['wav', 'mp3', 'm4a', 'flac', 'aac'],
            help="支持多种音频格式，建议文件大小不超过100MB"
        )
        
        # 新增热词输入框
        hotword = st.text_input(
            "热词（可选，多个词用空格分隔）",
            value="",
            help="可输入一组热词，提升特定词语识别准确率。多个词必须用空格分隔，如：词1 词2 词3"
        )
        
        # 处理按钮
        st.markdown("---")
        if st.button("🚀 开始转写", type="primary", use_container_width=True):
            if uploaded_file:
                # 传递热词参数
                process_audio(uploaded_file, hotword)
            else:
                st.warning("请上传音频文件")
    
    with col2:
        st.subheader("📝 转写结果")
        if 'transcribed_text' in st.session_state:
            text = st.session_state.transcribed_text
            st.text_area(
                "转写文本",
                value=text,
                height=400,
                help="转写结果将显示在这里，如果文本过长可以滚动查看。"
            )
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.download_button(
                    label="💾 下载为TXT",
                    data=text,
                    file_name="transcription.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_btn2:
                if st.button("🔄 清空结果", use_container_width=True):
                    clear_results()
        else:
            st.info("👆 请先上传音频文件并开始转写")

def qa_split_tab():
    """问答对拆分功能"""
    st.markdown('<h2 class="main-header">✂️ 问答对拆分</h2>', unsafe_allow_html=True)

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 问答拆分配置")
        custom_prompt = st.text_area(
            "问答提取Prompt",
            value=QA_EXTRACTION_PROMPT,
            height=300,
            help="你可以修改此Prompt来优化提取效果"
        )
        st.markdown("---")
        st.markdown("### 📊 拆分统计")
        if 'qa_pairs' in st.session_state:
            stats = st.session_state.get('qa_pairs', [])
            st.metric("问答对数量", len(stats))
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 原始文本")
        # 让文本框可编辑，不再依赖ASR页面的结果
        raw_text_placeholder = "请在此处粘贴或输入需要处理的文本..."
        raw_text = st.text_area(
            "待处理文本",
            value=st.session_state.get("qa_input_text", raw_text_placeholder),
            height=400,
            key="qa_input_text"
        )

        if st.button("🚀 开始提取", use_container_width=True, type="primary"):
            if raw_text == raw_text_placeholder or not raw_text.strip():
                st.warning("请输入要分析的文本。")
            else:
                # 清空之前的结果
                if 'qa_pairs' in st.session_state:
                    del st.session_state['qa_pairs']
                if 'raw_llm_output' in st.session_state:
                    del st.session_state['raw_llm_output']

                with st.spinner("🤖 正在调用Qwen大模型进行问答拆分..."):
                    try:
                        # 使用当前文本框内的内容进行流式处理
                        response_stream = get_qa_pairs_from_text_stream(raw_text, custom_prompt)
                        
                        # 在col2中显示流式输出
                        with col2:
                            st.subheader("🤖 问答对提取结果")
                            placeholder = st.empty()
                            full_response = placeholder.write_stream(response_stream)
                        
                        # 流结束后，解析完整内容并保存
                        qa_pairs = extract_qa_pairs_from_llm_result(full_response)
                        st.session_state.qa_pairs = qa_pairs
                        st.session_state.raw_llm_output = full_response
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 问答拆分失败: {e}")

    with col2:
        if 'qa_pairs' not in st.session_state:
             st.subheader("🤖 问答对提取结果")
             st.info("👆 在左侧输入文本，然后点击'开始提取'。")
        else:
            st.subheader("✅ 提取结果")
            qa_pairs = st.session_state.qa_pairs
            
            if not qa_pairs:
                st.warning("未能从文本中提取出任何问答对。")
                if 'raw_llm_output' in st.session_state:
                    with st.expander("查看LLM原始输出"):
                        st.code(st.session_state.raw_llm_output, language='json')
            else:
                df = pd.DataFrame(qa_pairs)
                
                st.dataframe(
                    df, 
                    use_container_width=True,
                    column_config={
                        "问题": st.column_config.TextColumn("问题", width="medium"),
                        "回答": st.column_config.TextColumn("回答", width="large"),
                    }
                )

                @st.cache_data
                def convert_df_to_csv(df_to_convert):
                    return df_to_convert.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(df)

                st.download_button(
                    label="📥 下载为CSV文件",
                    data=csv,
                    file_name="qa_pairs.csv",
                    mime="text/csv",
                    use_container_width=True
                )

def qa_smooth_tab():
    """问答对平顺功能"""
    st.markdown('<h2 class="main-header">✨ 问答对平顺</h2>', unsafe_allow_html=True)
    st.info("🔧 此功能正在开发中...")

def structured_output_tab():
    """结构化输出功能"""
    st.markdown('<h2 class="main-header">📊 结构化输出</h2>', unsafe_allow_html=True)
    st.info("🔧 此功能正在开发中...")

def main():
    """主函数"""
    st.markdown('<h1 class="main-header">🎤 音频文本处理系统</h1>', unsafe_allow_html=True)
    get_asr_model()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎤 ASR转写", 
        "✂️ 问答拆分", 
        "✨ 问答平顺", 
        "📊 结构化输出"
    ])
    
    with tab1:
        asr_tab()
    
    with tab2:
        qa_split_tab()
    
    with tab3:
        qa_smooth_tab()
    
    with tab4:
        structured_output_tab()

if __name__ == "__main__":
    main()
