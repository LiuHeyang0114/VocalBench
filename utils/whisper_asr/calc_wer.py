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
from shutil import copyfile

english_normalizer = EnglishTextNormalizer()

ordinal_words = {
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
    'fifth': 5,
    'sixth': 6,
    'seventh': 7,
    'eighth': 8,
    'ninth': 9,
    'tenth': 10,
    'eleventh': 11,
    'twelfth': 12,
    'thirteenth': 13,
    'fourteenth': 14,
    'fifteenth': 15,
    'sixteenth': 16,
    'seventeenth': 17,
    'eighteenth': 18,
    'nineteenth': 19,
    'twentieth': 20,
    '1st': 1,
    '2nd': 2,
    '3rd': 3,
    '4th': 4,
    '5th': 5,
    '6th': 6,
    '7th': 7,
    '8th': 8,
    '9th': 9,
    '10th': 10,
    '11st': 11,
    '11th': 11,
    '12th': 12,
    '12nd': 12,
    '13rd': 13,
    '13th': 13,
    '14th': 14,
    '15th': 15,
    '16th': 16,
    '17th': 17,
    '18th': 18,
    '19th': 19,
    '20th': 20,
}

def replace_ordinal_words(text):
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in ordinal_words.keys()) + r')\b'
    
    def replacer(match):
        word = match.group(0)
        return str(ordinal_words[word])
    
    result = re.sub(pattern, replacer, text, flags=re.IGNORECASE)
    return result

pattern = re.compile(
    rf'(\b(?:{"|".join(re.escape(k) for k in ordinal_words)})\b[,.]?)'   
    r'|'                                                                 
    r'(\b\d+(?:st|nd|rd|th)\b)',                                       
    re.IGNORECASE
)

def replace_ordinal(match):
    word_part = match.group(1)
    digit_part = match.group(2)

    if word_part:
        word = word_part.lower().strip(".,")
        if word in ordinal_words:
            return f"{ordinal_words[word]}."
    
    elif digit_part:
        number = re.match(r'\d+', digit_part).group()
        return f"{number}."

    return match.group(0)


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return []

def remove_sp(text, language):
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)
    gt = re.sub(f" ?([!,.?;:])", r"\1", gt)
    gt = re.sub(r"-", " ", gt)
    gt = gt.lstrip(" ")
    if language == "zh":
        gt = re.sub(rf"\s+", r"", gt)
    return gt

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
        pred = pattern.sub(replace_ordinal, pred)
        
        if language in ["en"]:
            ref = english_normalizer(ref)
            pred = english_normalizer(pred)
        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()

        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
        
    return distance, ref_length

def calc_wer(input_json):
    with open(input_json, 'r', encoding='utf-8') as infile:
      data = json.load(infile)
    language = 'en'
    
    instances = []
    total_distance = 0
    total_ref_length = 0
    for item in data:
      gt = remove_sp(item['prediction'], language)
      hyp = remove_sp(item['asr_pred'], language)
      distance, ref_length = compute_wer([gt], [hyp], language)
      instance_wer = distance / ref_length if ref_length != 0 else 0.0
      total_distance += distance
      total_ref_length += ref_length
      instances.append({
        "gt": gt,
        "hyp": hyp,
        "wer": instance_wer
      })
    
    total_wer = total_distance / total_ref_length *100 if total_ref_length != 0 else 0.0
    print(f"{total_wer:.6f}")


     
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str)
    args = parser.parse_args()

    calc_wer(args.input_json)
    
