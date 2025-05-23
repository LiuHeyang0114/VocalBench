import json 
import os 
import argparse
import editdistance as ed
from whisper_asr.evaluate_tokenizer import EvaluationTokenizer
from whisper_asr.whisper_normalizer.english import EnglishTextNormalizer
from funasr import AutoModel
import os
import json
import argparse
from pydub import AudioSegment
import pdb
import warnings
import sys

warnings.filterwarnings("ignore")
# sys.stderr = open('/dev/null', 'w')  


english_normalizer = EnglishTextNormalizer()

def compute_wer(refs, hyps, language):
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
        tokenizer_type="none",
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )
    for i in range(len(refs)):
        ref = refs[i]
        pred = hyps[i]
        if language in ["en"]:
            ref = english_normalizer(ref)
            pred = english_normalizer(pred)
        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()
        

        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
        
    return distance, ref_length

def find_emotion_id(text):
    id2emotion = ['angry', 'happy', 'neutral', 'sad', 'surprised', 'unknown']
    text_lower = text.lower() 
    for i, emotion in enumerate(id2emotion):
        if emotion in text_lower:
            return i
    return id2emotion.index('unknown') 


def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration = len(audio) 
    return duration

def count_words(text):
    return len(text.strip().split())

def gen_score(eval_json, only_score):
    wav_dir = eval_json.replace('result/instruction.json', 'wav/instruction')
    query_dir = '../audio/instruction_following'
    model_id = "../tools/emotion2vec_plus_large"
    emotion_model = AutoModel(
          model=model_id,
          hub="ms",  
      )

    sucess = 0
    with open(eval_json, "r") as f:
        data = json.load(f)
    
    num = len(data)
    for instance in data:
        index = int(instance['Qid'].split('-')[1])
        wav_file = os.path.join(wav_dir, str(index) + '_pred.wav')
        # wav_file = os.path.join(wav_dir, instance['Qid'] + '.wav')
        if index < 200:
          gt = instance["Question"].split('.', 1)[1].lower()
          hyp = instance["Prediction"].lower()
          distance, ref_length = compute_wer([gt], [hyp], 'en')
          instance_wer = distance / ref_length if ref_length != 0 else 0.0
          if index < 50:
            if (instance_wer) < 0.2:
              sucess+=instance["s2t_score"]
          elif index in range(50, 100):
            if (instance_wer) < 0.2:
              emo_id = find_emotion_id(instance['Question'])
              rec_result = emotion_model.generate(wav_file, output_dir="./save", granularity="utterance", extract_embedding=False)
              prob_scores = rec_result[0]['scores'][emo_id]
              if prob_scores > 0.5:
                sucess+=1
          elif index in range(100, 150):
            if (instance_wer) < 0.2:
              all_len = count_words(instance["Question"])
              valide_len = count_words(instance["Question"].split('.', 1)[1])
              query_dur = get_audio_duration(os.path.join(query_dir, instance['Qid'].split('-')[1] + '.wav'))* valide_len / all_len
              response_dur = get_audio_duration(wav_file) 
              if "twice" in instance["Question"].split('.', 1)[0].lower():
                min_dur = query_dur * 0.25
                max_dur = query_dur * 0.75
              elif "half" in instance["Question"].split('.', 1)[0].lower():
                min_dur = query_dur * 1.5
                max_dur = query_dur * 2.5
              if min_dur < response_dur < max_dur:
                sucess+=1       
          elif index in range(150, 200):
            if (instance_wer) < 0.2:
              emo_id = find_emotion_id(instance['Question'])
              rec_result = emotion_model.generate(wav_file, output_dir="./from_yuhao", granularity="utterance", extract_embedding=False)
              prob_scores = rec_result[0]['scores'][emo_id]
              if prob_scores > 0.5:
                all_len = count_words(instance["Question"])
                valide_len = count_words(instance["Question"].split('.', 1)[1])
                query_dur = get_audio_duration(os.path.join(query_dir, instance['Qid'].split('-')[1] + '.wav'))* valide_len / all_len
                response_dur = get_audio_duration(wav_file) 
                if "twice" in instance["Question"].split('.', 1)[0].lower():
                  min_dur = query_dur * 0.25
                  max_dur = query_dur * 0.75
                elif "half" in instance["Question"].split('.', 1)[0].lower():
                  min_dur = query_dur * 1.5
                  max_dur = query_dur * 2.5
                if min_dur < response_dur < max_dur:
                  sucess+=1
        elif index>=200:
          if index in range(500, 550):
            sucess+=instance["s2s_score"]
          else:
            sucess+=instance["s2t_score"]      
    if not only_score:   
      print(f"Total_num: {num}; Sucess_num: {sucess}")
    else:
      print(sucess/num*100)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", type=str)
    parser.add_argument("--only_score", action='store_true')
    args = parser.parse_args()

    gen_score(args.eval_json, args.only_score)
