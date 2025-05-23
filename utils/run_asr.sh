#!/bin/bash

eval_set=(
  single_round
)

# eval_set=(
#   knowledge
#   reasoning
#   creativity
#   single_round
#   multi_round
#   instruction
#   safety
#   emotion
#   robust
# )
eval_model='llama-omni'

mkdir -p ../result/${eval_model}/json_asr

for set in ${eval_set[@]}; do
  python3 whisper_asr/whisper_asr.py --input_json ../result/${eval_model}/json/${set}.json \
                                    --output_json ../result/${eval_model}/json_asr/${set}.json \
                                    --wav_dir ../result/${eval_model}/wav/${set}
done