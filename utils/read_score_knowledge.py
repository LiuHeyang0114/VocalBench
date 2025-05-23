import json
import re
from whisper_asr.whisper_normalizer.english import EnglishTextNormalizer
import argparse


source_to_id = {
    "llamaq-test": 0,
    "webq-test": 1,
    "triviaqa-dev": 2,
    "sciq-test": 3
}

set_to_id = {
    "Art": 0,
    "Biology": 1,
    "Celebrity": 2,
    "Chemistry": 3,
    "Economics": 4, 
    "Geography": 5,
    "History": 6,
    "Literature": 7,
    "Music": 8,
    "Physics": 9,
    "Psychology": 10,
    "Society": 11,
    "Sports": 12
}

id_to_source = {v: k for k, v in source_to_id.items()}
id_to_set = {v: k for k, v in set_to_id.items()}


def remove_sp(text):
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)
    gt = re.sub(f" ?([!,.?;:])", r"\1", gt)
    gt = re.sub(r"-", " ", gt)
    gt = gt.lstrip(" ")
    return gt

def eval_knowledge(eval_json, output_json, only_score):
    english_normalizer = EnglishTextNormalizer()
    ans_file = '../json/knowledge.json'
    with open(ans_file, 'r', encoding='utf-8') as ans:
      ans = json.load(ans)
    with open(eval_json, 'r', encoding='utf-8') as pred:
      data = json.load(pred)
    
    s2t_right = 0
    s2s_right = 0
    source_total = [0] * 4
    source_right_s2t = [0] * 4
    source_right_s2s = [0] * 4
    set_total = [0] * 13
    set_right_s2t = [0] * 13
    set_right_s2s = [0] * 13

    for item in data:
      index = int(item['question_id'].split('-')[1])
      source_id = source_to_id[ans[index]["Source"]]
      set_id = set_to_id[ans[index]["Topic"]]
      source_total[source_id] += 1
      set_total[set_id] += 1
      item_ans = english_normalizer(remove_sp(ans[index]["Answer"])).lower()
      pred = english_normalizer(remove_sp(item['prediction'])).lower()
      asr_pred = english_normalizer(remove_sp(item['asr_pred'])).lower()

      if item_ans in pred:
        s2t_right += 1
        source_right_s2t[source_id] += 1
        set_right_s2t[set_id] += 1
        item['s2t_right'] = 1
      if item_ans in asr_pred:
        s2s_right += 1
        source_right_s2s[source_id] += 1
        set_right_s2s[set_id] += 1
        item['s2s_right'] = 1

    if not only_score:
      print("s2t_acc:", s2t_right/len(data), "s2s_acc:", s2s_right/len(data))
      for i in range(len(source_total)):
        print("Source:", id_to_source[i], "  Total QA:", source_total[i], "  s2t  Accuracy:", source_right_s2t[i]/source_total[i]*100, "  s2s  Accuracy:", source_right_s2s[i]/source_total[i]*100)
      for i in range(len(set_total)):
        print("Set:", id_to_set[i], "  Total QA:", set_total[i], "  s2t  Accuracy:", set_right_s2t[i]/set_total[i]*100, "  s2s  Accuracy:", set_right_s2s[i]/set_total[i]*100)
    else:
      print(s2s_right/len(data))
    
    with open(output_json, 'w', encoding='utf-8') as output:
      json.dump(data, output, ensure_ascii=False, indent=4)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", type=str)
    parser.add_argument("--only_score", action='store_true')
    args = parser.parse_args()

    output_json = args.eval_json.replace('json_asr', 'result')
    eval_knowledge(args.eval_json, output_json, args.only_score)