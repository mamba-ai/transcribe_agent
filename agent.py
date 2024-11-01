import torch
import torchaudio
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path
from pydub import AudioSegment as PydubSegment
import os 
import uuid
import shutil

from math import floor
from typing import Optional, List, Dict, Any
import torch
import gradio as gr
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read


# configuration
MODEL_NAME = "federerjiang/mambavoice-ja-v1"
BATCH_SIZE = 4
CHUNK_LENGTH_S = 15
FILE_LIMIT_MB = 1000
TOKEN = os.environ.get('HF_TOKEN', None)
# device setting
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda:0"
else:
    torch_dtype = torch.float32
    device = "cpu"

# define the pipeline
pipe = pipeline(
    model=MODEL_NAME,
    chunk_length_s=CHUNK_LENGTH_S,
    batch_size=BATCH_SIZE,
    torch_dtype=torch_dtype,
    device=device,
    trust_remote_code=True,
    token=TOKEN,
)


@dataclass
class SilenceSegment:
    start: float
    end: float
    duration: float
    speech_prob: float

class SilenceDetector:
    def __init__(
        self,
        threshold: float = 0.3,
        min_silence_duration: float = 0.5,
        sampling_rate: int = 16000
    ):
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.sampling_rate = sampling_rate
        self.window_size_samples = 512 if sampling_rate == 16000 else 256
        
        # 加载模型
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True
        )
        self.model.eval()
    
    def detect_silence(self, audio_path: str) -> Tuple[List[SilenceSegment], List[float], List[float]]:
        # 加载音频
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            wav = resampler(wav)
        
        # 确保音频是单声道
        if wav.dim() > 2:
            wav = wav.mean(dim=0)
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        
        # 计算语音概率
        speech_probs = []
        timestamps = []
        current_time = 0.0
        window_size = self.window_size_samples
        
        for i in range(0, len(wav), window_size):
            segment = wav[i:i + window_size]
            if len(segment) < window_size:
                segment = torch.nn.functional.pad(
                    segment, 
                    (0, window_size - len(segment))
                )
            
            # 正确设置维度 [1, samples]
            segment = segment.unsqueeze(0)
            
            # 获取语音概率
            with torch.no_grad():
                speech_prob = self.model(segment, self.sampling_rate).item()
            
            speech_probs.append(speech_prob)
            timestamps.append(current_time)
            current_time += window_size / self.sampling_rate
        
        # 找出静默片段
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, (time, prob) in enumerate(zip(timestamps, speech_probs)):
            if not in_silence and prob < self.threshold:
                silence_start = time
                in_silence = True
            elif in_silence and (prob >= self.threshold or i == len(timestamps) - 1):
                silence_duration = time - silence_start
                if silence_duration >= self.min_silence_duration:
                    segment = SilenceSegment(
                        start=silence_start,
                        end=time,
                        duration=silence_duration,
                        speech_prob=np.mean(speech_probs[
                            int(silence_start * self.sampling_rate / window_size):
                            int(time * self.sampling_rate / window_size)
                        ])
                    )
                    silence_segments.append(segment)
                in_silence = False
        
        return silence_segments, speech_probs, timestamps
    

@dataclass
class Segment:    # 改名为 Segment
    start: float
    end: float
    duration: float

class AudioSplitter:
    def __init__(
        self,
        min_segment_length: float = 10.0,
        max_segment_length: float = 30.0,
        vad_threshold: float = 0.3,
        min_silence_duration: float = 0.5,
        padding_duration: float = 0.5
    ):
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.padding_duration = padding_duration
        
        # 初始化静默检测器
        self.silence_detector = SilenceDetector(
            threshold=vad_threshold,
            min_silence_duration=min_silence_duration
        )
    
    def find_best_split_points(
        self,
        silence_segments: List[SilenceSegment],
        total_duration: float
    ) -> List[Segment]:
        segments = []
        current_start = 0.0
        
        while current_start < total_duration:
            best_end = None
            best_silence = None
            
            for silence in silence_segments:
                if silence.start <= current_start:
                    continue
                
                potential_duration = silence.start - current_start
                
                if (self.min_segment_length <= potential_duration <= self.max_segment_length):
                    if best_end is None or potential_duration > best_end - current_start:
                        best_end = silence.start
                        best_silence = silence
            
            if best_end is None:
                if total_duration - current_start <= self.max_segment_length:
                    best_end = total_duration
                else:
                    best_end = current_start + self.max_segment_length
            
            segment = Segment(
                start=current_start,
                end=best_end,
                duration=best_end - current_start
            )
            segments.append(segment)
            
            if best_silence:
                current_start = best_silence.end
            else:
                current_start = best_end
        
        return segments

    def add_silence_padding(
        self,
        audio_segment: PydubSegment,
        sample_rate: int = 16000
    ) -> PydubSegment:
        """添加首尾静默片段"""
        padding_ms = int(self.padding_duration * 1000)
        # 创建静默片段
        silence = PydubSegment.silent(
            duration=padding_ms,
            frame_rate=sample_rate
        )
        return silence + audio_segment + silence
    
    def split_audio(
        self,
        input_path: str,
        output_dir: str = "segments"
    ) -> List[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 检测静默片段
        silence_segments, _, _ = self.silence_detector.detect_silence(input_path)
        
        # 使用 pydub.AudioSegment 加载音频
        audio = PydubSegment.from_file(input_path)
        # 获取采样率
        sample_rate = audio.frame_rate
        total_duration = len(audio) / 1000.0
        
        # 找到最佳分割点
        segments = self.find_best_split_points(silence_segments, total_duration)
        
        # 分割并保存音频
        output_files = []
        for i, segment in enumerate(segments):
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            
            audio_segment = audio[start_ms:end_ms]

            # 添加静默片段
            padded_segment = self.add_silence_padding(
                audio_segment, 
                sample_rate=sample_rate
            )
            
            output_path = output_dir / f"segment_{i:03d}.wav"
            padded_segment.export(str(output_path), format="wav")
            output_files.append(str(output_path))
            
            print(f"Saved segment {i}: {segment.duration:.2f}s "
                  f"({segment.start:.2f}s - {segment.end:.2f}s)")
        return output_files
    

    def analyze_segments(self, segments: List[Segment]):
        durations = [seg.duration for seg in segments]
        
        print("\nSegment Analysis:")
        print(f"Total segments: {len(segments)}")
        print(f"Average duration: {np.mean(durations):.2f}s")
        print(f"Min duration: {np.min(durations):.2f}s")
        print(f"Max duration: {np.max(durations):.2f}s")
        
        invalid_segments = [
            (i, seg) for i, seg in enumerate(segments)
            if seg.duration < self.min_segment_length or seg.duration > self.max_segment_length
        ]
        
        if invalid_segments:
            print("\nWarning: Found invalid segments:")
            for i, seg in invalid_segments:
                print(f"Segment {i}: {seg.duration:.2f}s")


def get_prediction(inputs, prompt: Optional[str]=None):
    generate_kwargs = {
        "language": "japanese", 
        "task": "transcribe",
        "length_penalty": 0,
        "num_beams": 2,
    }
    if prompt:
        generate_kwargs['prompt_ids'] = pipe.tokenizer.get_prompt_ids(prompt, return_tensors='pt').to(device)
    prediction = pipe(inputs, return_timestamps=True, generate_kwargs=generate_kwargs)
    text = "".join([c['text'] for c in prediction['chunks']])
    return text


def generate_random_folder():
    """Generate a random folder name using UUID"""
    return str(uuid.uuid4())


def transcribe(inputs: str):
    if inputs is None:
        raise gr.Error("音声ファイルが送信されていません！リクエストを送信する前に、音声ファイルをアップロードまたは録音してください。")
    splitter = AudioSplitter(
        min_segment_length=2.0,
        max_segment_length=10.0,
        vad_threshold=0.3,
        min_silence_duration=0.5,
        padding_duration=0.5
    )

    folder_name = generate_random_folder()
    output_dir = f"temp/{folder_name}"
    output_files = splitter.split_audio(inputs, output_dir=output_dir)

    outputs = []
    for output_file in output_files:
        with open(output_file, "rb") as f:
            inputs = f.read()
            inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
            inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}
            outputs.append(get_prediction(inputs))
    
    shutil.rmtree(output_dir)
    return "".join(outputs)


# 使用示例
def main():
    splitter = AudioSplitter(
        min_segment_length=2.0,
        max_segment_length=10.0,
        vad_threshold=0.3,
        min_silence_duration=0.5,
        padding_duration=0.5
    )
    
    AUDIO_PREFIXs = [
        "C773501-D20211130114458-T114458-P_I-K0452228282-F1-94",
        "C773501-D20211130160855-T160855-P_I-K09091525469-F1-89",
        "C773501-D20220311101913-T101913-P_I-K0452228282-F1-11"
    ]
    for audio_prefix in AUDIO_PREFIXs:
        input_file = f"assets/{audio_prefix}.wav"
        output_dir = f"output/{audio_prefix}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_files = splitter.split_audio(input_file, output_dir)
        print(f"\nCreated {len(output_files)} segments:")
        for file in output_files:
            print(f"- {file}")

if __name__ == "__main__":
    main()