import sys
import json 
import random 
import traceback 
import base64
import os 
import pdb
import requests
import re
import argparse
from glob import glob
from funasr import AutoModel
import numpy as np

EMOTION = '../json/emotion.json'

def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum(axis=0)

def inverse_softmax(p):
    if np.any(p <= 0):
        raise ValueError("Probability contains 0")
    return np.log(p)

def emotion_acoustic(wav_dir, output_json):
    with open(EMOTION, 'r') as f:
      metadata = json.load(f)
    wav_list = glob(os.path.join(wav_dir, "*.wav"))

    acoustic_score = []
    model_id = "../tools/emotion2vec_plus_large"
    emotion_model = AutoModel(
        model=model_id,
        hub="ms",  
    )

    total_score = 0
    failed = 0
    for wav_file in wav_list:
      index = int(re.findall(r'\d+', wav_file.split('/')[-1])[0])
      scores = metadata[index]['Score']
      try:
        rec_result = emotion_model.generate(wav_file, output_dir="./save", granularity="utterance", extract_embedding=False)[0]['scores']
        logits = inverse_softmax(np.array(rec_result))
        logits_top5 = logits[:-1]
        rec_result = softmax(logits_top5)

        result = sum(a * b for a, b in zip(rec_result, scores))
        acoustic_score.append({
          'Qid': wav_file.split('/')[-1].split('.')[0],
          'Response Emotion': rec_result.tolist(),
          'Emotion Score': scores,
          'Acoustic_score': result
        })
        total_score = total_score + result 
      except:
        failed+=1
      
      with open(output_json, 'w+', encoding = 'utf-8') as outf:
        json.dump(acoustic_score, outf, ensure_ascii=False, indent=4)
    
    print(f"Process {len(wav_list)-failed} files, avg score is {total_score/(len(wav_list)-failed)}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str)
    args = parser.parse_args()

    output_json = args.wav_dir.replace('wav/emotion', 'result/emotion_acoustic_norm.json')
    emotion_acoustic(args.wav_dir, output_json)
