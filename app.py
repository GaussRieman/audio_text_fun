import streamlit as st
import os
import time
import json
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from qwen_llm import process_text_with_qwen, extract_qa_pairs_from_llm_result, split_text_to_qa_pairs

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

# 初始化ASR模型
@st.cache_resource
def initialize_asr_model():
    """初始化ASR模型（应用启动时调用）"""
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={
                "max_single_segment_time": 60000,
                "min_single_segment_time": 1000,
                "max_segment_length": 100000,
            },
            device=device,
            disable_update=True,  # 禁止联网检查和下载
        )
        return model, device, None
    except Exception as e:
        return None, device, str(e)

def get_asr_model():
    """获取ASR模型实例"""
    if 'asr_model' not in st.session_state:
        model, device, err = initialize_asr_model()
        st.session_state['asr_model'] = model
        st.session_state['asr_device'] = device
        st.session_state['asr_error'] = err
    return st.session_state.get('asr_model'), st.session_state.get('asr_device'), st.session_state.get('asr_error')

def asr_tab():
    """ASR音频转写功能"""
    st.markdown('<h2 class="main-header">🎤 音频转写 (ASR)</h2>', unsafe_allow_html=True)
    model, device, err = get_asr_model()
    # 只在首次加载成功时弹出toast
    if err:
        st.error(f"❌ ASR模型加载失败: {err}")
        return
    if not model:
        st.info("🔄 正在初始化ASR模型，请稍候...")
        return
    # 只弹一次toast
    if not st.session_state.get("asr_model_loaded_toast", False):
        if hasattr(st, "toast"):
            st.toast(f"✅ ASR模型加载成功！使用设备: {device}", icon="✅")
        else:
            st.success(f"✅ ASR模型加载成功！使用设备: {device}")
        st.session_state["asr_model_loaded_toast"] = True
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ ASR配置")
        
        # 显示模型信息
        st.markdown("### 📋 模型信息")
        st.info(f"设备: {'GPU' if 'cuda' in device else 'CPU'}")
        st.info("模型: SenseVoiceSmall")
        
        # 参数配置
        st.markdown("### ⚙️ 参数配置")
        batch_size = st.slider("批处理大小", min_value=60, max_value=200, value=120, step=20)
        merge_length = st.slider("合并长度(秒)", min_value=10, max_value=60, value=30, step=5)
        
        st.markdown("---")
        st.markdown("### 📊 处理统计")
        # 统计每次都从 session_state 读取，保证自动刷新
        stats = st.session_state.get('asr_stats', {})
        st.metric("处理时间", f"{stats.get('time', 0):.2f}秒")
        st.metric("文本长度", f"{stats.get('text_length', 0)}字符")
    
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
        
        # 处理按钮
        st.markdown("---")
        if st.button("🚀 开始转写", type="primary", use_container_width=True):
            if uploaded_file:
                process_audio(uploaded_file, batch_size, merge_length)
            else:
                st.warning("请上传音频文件")
    
    with col2:
        st.subheader("📝 转写结果")
        
        # 显示转写结果
        if 'transcribed_text' in st.session_state:
            text = st.session_state.transcribed_text
            
            # 文本显示
            st.text_area(
                "转写文本",
                value=text,
                height=400,
                help="转写结果将显示在这里"
            )
            
            # 操作按钮
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                if st.button("📋 复制文本", use_container_width=True):
                    st.write("文本已复制到剪贴板")
                    st.session_state['copied'] = True
            
            with col_btn2:
                # 下载为txt
                st.download_button(
                    label="💾 下载为TXT",
                    data=text,
                    file_name="transcription.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col_btn3:
                if st.button("🔄 清空结果", use_container_width=True):
                    clear_results()
            
            # 显示复制成功消息
            if st.session_state.get('copied', False):
                st.success("✅ 文本已复制到剪贴板")
                st.session_state['copied'] = False
        
        else:
            st.info("👆 请先上传音频文件并开始转写")

def process_audio(uploaded_file, batch_size, merge_length):
    """处理音频文件"""
    try:
        model, device, err = get_asr_model()
        if err:
            st.error(f"❌ ASR模型加载失败: {err}")
            return
        if not model:
            st.error("ASR模型未加载")
            return
        
        # 确定音频文件路径
        audio_path = None
        if uploaded_file:
            # 保存上传的文件
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            audio_path = os.path.join(temp_dir, uploaded_file.name)
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            st.error("无效的音频文件路径")
            return
        
        # 开始转写
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🎵 正在转写音频...")
        progress_bar.progress(25)
        
        start_time = time.time()
        
        # 执行转写
        res = model.generate(
            input=audio_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=batch_size,
            merge_vad=True,
            merge_length_s=merge_length,
        )
        
        progress_bar.progress(75)
        status_text.text("📝 正在后处理文本...")
        
        # 后处理
        transcribed_text = rich_transcription_postprocess(res[0]["text"])
        
        progress_bar.progress(100)
        status_text.text("✅ 转写完成!")
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 保存结果到session state
        st.session_state.transcribed_text = transcribed_text
        st.session_state.asr_stats = {
            'time': processing_time,
            'text_length': len(transcribed_text),
            'file_name': uploaded_file.name
        }
        
        # 显示成功消息
        st.success(f"🎉 转写完成！耗时 {processing_time:.2f} 秒")
        
        # 清理临时文件
        if uploaded_file and os.path.exists(audio_path):
            os.remove(audio_path)
        # 兼容新旧Streamlit的自动刷新
        try:
            st.rerun()
        except AttributeError:
            try:
                st.experimental_rerun()
            except AttributeError:
                pass  # 低版本不支持自动刷新
        
    except Exception as e:
        st.error(f"❌ 转写失败: {str(e)}")
        st.exception(e)

def save_transcription_result(text):
    """保存转写结果"""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"transcription_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        
        st.success(f"✅ 结果已保存到: {filename}")
        
        # 同时保存JSON格式
        json_filename = f"transcription_{timestamp}.json"
        result_data = {
            "timestamp": timestamp,
            "text": text,
            "text_length": len(text),
            "stats": st.session_state.get('asr_stats', {})
        }
        
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        st.error(f"保存失败: {e}")

def clear_results():
    """清空结果"""
    if 'transcribed_text' in st.session_state:
        del st.session_state.transcribed_text
    if 'asr_stats' in st.session_state:
        del st.session_state.asr_stats
    st.rerun()

def qa_split_tab():
    """问答对拆分功能"""
    st.markdown('<h2 class="main-header">✂️ 问答对拆分</h2>', unsafe_allow_html=True)
    st.info("🔧 此功能正在开发中...")

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
    # 应用启动时初始化模型
    get_asr_model()
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎤 ASR转写", 
        "✂️ 问答拆分", 
        "✨ 问答平顺", 
        "📊 结构化输出"
    ])
    
    # 标签页内容
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
