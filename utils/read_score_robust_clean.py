import json
import argparse

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_qids_by_answer(input_json, refer_json):
    input_answers = {item['Answer'] for item in input_json}

    matched_qids = []
    for item in refer_json:
        if item['Answer'] in input_answers:
            matched_qids.append(item['Qid'])
    
    return matched_qids

def calculate_average_score(result_json, qid_list):
    scores = []
    for item in result_json:
        if item['Qid'] in qid_list and 's2s_score' in item:
            scores.append(item['s2s_score'])

    if not scores:
        return 0.0
    return sum(scores) / len(scores)

if __name__ == '__main__':
    robust_clean_path = '../json/robust_clean.json'
    refer_path = '../json/single_round.json'
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_json", type=str)
    args = parser.parse_args()

    input_data = load_json(robust_clean_path)
    refer_data = load_json(refer_path)
    result_data = load_json(args.result_json)
    matched_qids = find_qids_by_answer(input_data, refer_data)

    average_score = calculate_average_score(result_data, matched_qids)
    print(average_score)
