import os
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

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
        if hotword:
            generate_kwargs["hotword"] = hotword
        generate_kwargs.update(kwargs)
        res = self.model.generate(**generate_kwargs)
        text = rich_transcription_postprocess(res[0]["text"])
        return text