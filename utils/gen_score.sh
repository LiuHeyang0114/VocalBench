#!/bin/bash

eval_model='llama-omni'

knowledge=$(python3 read_score_knowledge.py --eval_json ../result/${eval_model}/json_asr/knowledge.json --only_score| awk '{printf "%.2f", $0 * 100}')
reasoning=$(python3 read_score.py --eval_json ../result/${eval_model}/result/reasoning.json --only_score)
creativity=$(python3 read_score.py --eval_json ../result/${eval_model}/result/creativity.json --only_score)
single_round=$(python3 read_score.py --eval_json ../result/${eval_model}/result/single_round.json --only_score)
multi_round=$(python3 read_score.py --eval_json ../result/${eval_model}/result/multi_round.json --only_score)
instruction=$(python3 read_score_instruction.py --eval_json ../result/${eval_model}/result/instruction.json --only_score 2>/dev/null | awk '
{
    for (i=1; i<=NF; i++) {
        if ($i ~ /^[+-]?[0-9]*\.?[0-9]+$/) {
            val=$i;
        }
    }
}
END { if (val != "") print val; else print "0"; }')
emotion_acoustic=$(python3 read_score.py --eval_json ../result/${eval_model}/result/emotion_acoustic_norm.json --only_score)
emotion_semantic=$(python3 read_score.py --eval_json ../result/${eval_model}/result/emotion.json --only_score)
safety=$(python3 read_score.py --eval_json ../result/${eval_model}/result/safety.json --only_score)
robust_clean=$(python3 read_score_robust_clean.py --result_json ../result/${eval_model}/result/single_round.json)
robustness=$(python3 read_score_robust.py --eval_json ../result/${eval_model}/result/robust.json --clean_score $robust_clean --only_score)
utmos=$(python3 avg_csv.py --eval_csv ../result/${eval_model}/result/chat_utmos.csv)
wer=$(python3 whisper_asr/calc_wer.py --input_json ../result/${eval_model}/json_asr/single_round.json)
wer_score=$(echo "$wer" | awk '{
    wer = $1;
    if (wer < 2) score = 10;
    else if (wer < 4) score = 9;
    else if (wer < 6) score = 8;
    else if (wer < 8) score = 7;
    else if (wer < 10) score = 6;
    else if (wer < 12) score = 5;
    else if (wer < 14) score = 4;
    else if (wer < 16) score = 3;
    else if (wer < 18) score = 2;
    else score = 1;
    print score;
}')
echo "knowledge: ${knowledge}"
echo "reasoning: ${reasoning}"
echo "creativity: ${creativity}"
echo "single_round: ${single_round}"
echo "multi_round: ${multi_round}"
echo "instruction: ${instruction}"
echo "emotion_acoustic: ${emotion_acoustic}"
echo "emotion_semantic: ${emotion_semantic}"
echo "safety: ${safety}"
echo "robustness: ${robustness}"
echo "wer_score: ${wer_score}"
echo "utmos: ${utmos}"

all_score=$(echo "scale=3; (${knowledge}/10 + ${reasoning}*2 + ${creativity}*2 + $wer_score + $utmos + ${single_round}*2 + ${instruction}/10 + $emotion_acoustic + $emotion_semantic +${safety}*10 + $robustness)" | bc)
echo $all_score