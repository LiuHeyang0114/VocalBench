#!/bin/bash

eval_set=(
  reasoning,
  creativity,
  single_round,
  multi_round,
  instruction,
  safety,
  emotion,
  robust
)

eval_model='llama-omni'
mkdir -p ../result/${eval_model}/result

for set in ${eval_set[@]}; do
  python3 qwenmax_eval.py --input_json ../result/${eval_model}/json_asr/${set}.json \
                          --output_json ../result/${eval_model}/result/${set}.json
done

python3 calc_emotion_acoustic.py --wav_dir ../result/${eval_model}/wav/emotion

python3 utmos_utils/predict.py --mode predict_dir --inp_dir ../result/${eval_model}/wav/single_round --bs 4 --out_path ../result/${eval_model}/result/chat_utmos.csv