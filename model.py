import os
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import re

class SenseVoiceModel:
    def __init__(self, model_dir="iic/SenseVoiceSmall", device=None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModel(
            model=model_dir,
            vad_model="fsmn-vad",
            vad_kwargs={
                "max_single_segment_time": 60000,
                "min_single_segment_time": 1000,
                "max_segment_length": 100000,
            },
            device=device,
            disable_update=True,
        )

    def transcribe(self, audio_file_path, language="auto", use_itn=True, batch_size_s=120, merge_vad=True, merge_length_s=30):
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_file_path}")
        res = self.model.generate(
            input=audio_file_path,
            cache={},
            language=language,
            use_itn=use_itn,
            batch_size_s=batch_size_s,
            merge_vad=merge_vad,
            merge_length_s=merge_length_s,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return text 
    


class ParaformerModel:
    def __init__(self, model_dir="paraformer-zh", device=None, use_vad=True, use_punc=True, use_spk=False):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        kwargs = {"model": model_dir, "device": device}
        if use_vad:
            kwargs["vad_model"] = "fsmn-vad"
        if use_punc:
            kwargs["punc_model"] = "ct-punc"
        if use_spk:
            kwargs["spk_model"] = "cam++"
        self.model = AutoModel(**kwargs)

    def transcribe(self, audio_file_path, batch_size_s=300, hotword=None, **kwargs):
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_file_path}")
        generate_kwargs = {"input": audio_file_path, "batch_size_s": batch_size_s}
        generate_kwargs['hotword'] = hotword
        generate_kwargs.update(kwargs)
        print("generate_kwargs: ", generate_kwargs)
        res = self.model.generate(**generate_kwargs)
        text = rich_transcription_postprocess(res[0]["text"])
        return text


def test_hotword_functionality(audio_file_path, hotword_str=None):
    """
    测试热词功能的本地验证函数
    
    Args:
        audio_file_path (str): 音频文件路径
        hotword_str (str, optional): 热词字符串，用逗号分隔
    
    Returns:
        dict: 包含转写结果和统计信息的字典
    """
    import time
    
    print(f"=== 热词功能测试 ===")
    print(f"音频文件: {audio_file_path}")
    print(f"热词: {hotword_str if hotword_str else '无'}")
    print("-" * 50)
    
    try:
        # 初始化模型
        print("正在加载模型...")
        model = ParaformerModel()
        print(f"模型加载完成，设备: {model.device}")
        
        # 执行转写
        print("开始转写...")
        start_time = time.time()
        
        transcribed_text = model.transcribe(audio_file_path, hotword=hotword_str)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 统计信息
        result = {
            'audio_file': audio_file_path,
            'hotword': hotword_str,
            'transcribed_text': transcribed_text,
            'processing_time': processing_time,
            'text_length': len(transcribed_text),
            'success': True
        }
        
        print(f"转写完成！")
        print(f"处理时间: {processing_time:.2f}秒")
        print(f"文本长度: {len(transcribed_text)}字符")
        print(f"转写结果: {transcribed_text}")
        
        return result
        
    except Exception as e:
        print(f"转写失败: {str(e)}")
        return {
            'audio_file': audio_file_path,
            'hotword': hotword_str,
            'error': str(e),
            'success': False
        }


if __name__ == "__main__":
    
    # 执行测试
    audio_path = "/home/frank/codes/audio_anything/audio_text_fun/assets/vad_example.wav"
    hotword = "是错 试验"
    result = test_hotword_functionality(audio_path, hotword)
    
    if result['success']:
        print("\n✅ 测试成功！")
    else:
        print("\n❌ 测试失败！")