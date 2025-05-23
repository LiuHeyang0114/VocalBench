import json

input_json = '/nas_works/467836/VocalBench/json/robust.json'
output_json = '/nas_works/467836/VocalBench/json/robust_new.json'

with open(input_json, 'r', encoding = 'utf-8') as f:
  data = json.load(f)

instances = []
for item in data:
  # instances.append({
  #       "Set": "Creativity",
  #       "Qid": item["Qid"],
  #       "Audio": item["Audio"],
  #       "Question": item["Question"],
  #       "Answer": item["Answer"],
  #       "Category": item["Category"],
  #       "Source": item["Source"]
  # })
  item['Set'] = item['Qid'].split('-')[1]
  if item['Set'] == 'distortion':
    item['Set'] = 'clipping_distortion'

with open(output_json, 'w+', encoding = 'utf-8') as outf:
  json.dump(data, outf, ensure_ascii=False, indent=4)