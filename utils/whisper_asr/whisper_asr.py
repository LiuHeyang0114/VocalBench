import os
import json
import re
from collections import defaultdict
import editdistance as ed
from evaluate_tokenizer import EvaluationTokenizer
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf 
import json
import os 
import numpy as np
import argparse
import pdb
import sys
import librosa
from shutil import copyfile

# EVAL_SETS = ["single_round", "creativity", "instruction", "knowledge", "multi_round", "reasoning", "safety", "emotion", "robust"]
# EVAL_MODELS = ["mini-omni", "mini-omni2", "llama-omni", "freeze-omni", "baichuan-omni", "minicpm-o", "vocalnet", "glm-4-voice", "qwen2.5-omni", "kimi-audio", "slam-omni"]


def load_whisper():
    local_model_path = "../tools/whisper" 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        local_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(local_model_path)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe


def remove_sp(text, language):
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)
    gt = re.sub(f" ?([!,.?;:])", r"\1", gt)
    gt = re.sub(r"-", " ", gt)
    gt = gt.lstrip(" ")
    if language == "zh":
        gt = re.sub(rf"\s+", r"", gt)
    return gt


def get_wave_file(eval_set, eval_model, index, qid):
  return f"{qid}.wav"
  # if eval_model == 'llama-omni':
  #   return f"{str(index)}_pred.wav"
  # elif eval_model in ['mini-omni', 'mini-omni2','baichuan-omni', 'freeze-omni', 'glm-4-voice', 'minicpm-o', 'qwen2.5-omni', 'kimi-audio', 'slam-omni']:
  #   return f"{qid}.wav"
  # elif eval_model in ['vocalnet']:
  #   return f"{index}.wav"
  # else:
  #   return None

def whisper_asr(input_json, output_json, wav_dir, eval_set, eval_model):
    language = 'en'
    asr_pipe = load_whisper()
    english_normalizer = EnglishTextNormalizer()
    with open(input_json, 'r', encoding='utf-8') as infile:
      data = [json.loads(line) for line in infile]
    
    instances = []
    index = 0
    for item in data:
        # index = item['question_id'].split('-')[1]
        qid = item['question_id']
        wav_path = os.path.join(wav_dir, get_wave_file(eval_set, eval_model, index, qid))
        if os.path.exists(wav_path):
          audio_input, sample_rate = sf.read(wav_path)
          sample = {"raw": audio_input, "sampling_rate": sample_rate}
          result = asr_pipe(sample)
          asr_pred = english_normalizer(remove_sp(result["text"], language))
          print(asr_pred)
          instances.append({
            "question_id": item['question_id'],
            "prediction": item['prediction'],
            "asr_pred": asr_pred
          })
        else:
          instances.append({
            "question_id": item['question_id'],
            "prediction": item['prediction'],
            "asr_pred": ""
          })
        index += 1
    
    with open(output_json, 'w+', encoding='utf-8') as f:
      json.dump(instances, f, ensure_ascii=False, indent=4)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str)
    parser.add_argument("--output_json", type=str)
    parser.add_argument("--wav_dir", type=str)
    args = parser.parse_args()
    
    eval_set = args.input_json.split('/')[-1].split('.')[0]
    eval_model = args.input_json.split('/')[-3]
    whisper_asr(args.input_json, args.output_json, args.wav_dir, eval_set, eval_model)
