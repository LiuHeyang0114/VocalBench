import sys
import json 
import random 
import traceback 
# from utils.whispermodel import WhisperModel
import base64
import os 
import pdb
import requests
import re
import argparse


def gen_score(eval_json, only_score):
    if 'emotion_acoustic' not in eval_json:
        total_s2t_score = 0
        total_s2s_score = 0
        with open(eval_json, "r") as f:
            data = json.load(f)
        num = len(data)
        for instance in data:
                total_s2t_score = total_s2t_score + instance["s2t_score"]
                total_s2s_score = total_s2s_score + instance["s2s_score"]
        if not only_score:
            print(f"Total Instances: {num}; Average s2t score: {total_s2t_score/num}; Average_s2s_score: {total_s2s_score/num}")
        else:
            print(total_s2s_score/num)
    else:
        total_acoustic_score = 0
        with open(eval_json, "r") as f:
            data = json.load(f)
        num = len(data)
        for instance in data:
                total_acoustic_score = total_acoustic_score + instance["Acoustic_score"]
        if not only_score:
            print(f"Total Instances: {num}; Average_s2s_score: {total_acoustic_score/num}")
        else:
            print(total_acoustic_score/num)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", type=str)
    parser.add_argument("--only_score", action='store_true')
    args = parser.parse_args()
    gen_score(args.eval_json, args.only_score)
