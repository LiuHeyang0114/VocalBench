import json
from collections import defaultdict
import argparse


def gen_score(file_path, clean_score, only_score):
    set_scores = defaultdict(lambda: {'s2t': [], 's2s': []})
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        qid = item.get('Qid')
        s2t_score = item.get('s2t_score')
        s2s_score = item.get('s2s_score')
        if not qid or s2t_score is None or s2s_score is None:
            continue  
        parts = qid.rsplit('-', 1)
        if len(parts) < 2:
            continue 
        set_name = parts[0]
        set_scores[set_name]['s2t'].append(s2t_score)
        set_scores[set_name]['s2s'].append(s2s_score)
    
    if not only_score:
        print("SetName\t\t\ts2t AvgScore\t\ts2s AvgScore")
        for set_name, scores in set_scores.items():
            avg_s2t = sum(scores['s2t']) / len(scores['s2t']) if scores['s2t'] else 0
            avg_s2s = sum(scores['s2s']) / len(scores['s2s']) if scores['s2s'] else 0
            print(f"{set_name}\t\t{avg_s2t:.4f}\t\t{avg_s2s:.4f}")
    else:
        eval_sets = ['robust-background_noise-snr_-5', 'robust-white_noise-snr_-5', 'robust-reverberation-rt60_30', 'robust-packet_loss-dropping_70', 'robust-farfield-filter_400hz', 'robust-distortion-clipping_0.0001']
        score = 0
        for set_name, scores in set_scores.items():
            if set_name in eval_sets:
                avg_s2s = sum(scores['s2s']) / len(scores['s2s']) if scores['s2s'] else 0
                pr = min(1, avg_s2s/clean_score)*2.5
                score = score + pr
        print(score)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", type=str)
    parser.add_argument("--clean_score", type=float, default=0.0)
    parser.add_argument("--only_score", action='store_true')
    args = parser.parse_args()

    gen_score(args.eval_json, args.clean_score, args.only_score)


    