<div align="center">


# **📈VocalBench: Benchmarking the Vocal Conversational Abilities for Speech Interaction Models**
<!-- # 🎧VocalNet: Speech LLM with Multi-Token Prediction for Faster and High-Quality Generation -->


</div>

## 👀 VocalBench Overview

**VocalBench** is a comprehensive evaluation benchmark to assess the vocal communication ability for speech interaction models.  

- **Semantic**: Abilities to generate accurate and vivid semantics, including knowledge, reasoning, and creativity sets.
- **Acoustic**: Speech response with spontaneous and natural acoustics, evaluated on the single-round set in the chat dimension.
- **Chat**: Performance on efficient and smooth chat, consisting of single- and multi-round, instruction following, emotional sympathy, safety alignment sets, and a real-time factor calculation representing the computing latency.
- **Robust**: Robustness under diverse acoustic environments, performing on white noise, background noise, reverberation, far-field, packet loss, and clipping distortion.

## 🙌 Quick Start

### Step 0: Model Preparation

- Environment Preparation:
```bash
cd tools/emotion2vec_plus_large
huggingface-cli download emotion2vec/emotion2vec_plus_large --local-dir .
echo -e "angry\nunuse_0\nunuse_1\nhappy\nneutral\nunuse_2\nsad\nsurprised\n<unk>" > tokens.txt
cd ../whisper
huggingface-cli download openai/whisper-large-v3 --local-dir .
cd ../
git clone https://huggingface.co/spaces/sarulab-speech/UTMOS-demo
cd UTMOS-demo
pip install -r requirements.txt
ln -s fairseq ../../utils/utmos_utils
cd ../..
pip install -r requirements.txt
```

### Step 1: Speech Interaction Model Infer

Example inference result: result/llama-omni

### Step 2: Perform Whisper ASR
```bash
cd utils
./run_asr.sh
```
### Step 3: Perform Qwen2.5-Max Eval & Emotion Acoustic Eval
```bash
cd utils
./run_qwenmax_eval.sh
```
### Step 4: Read Score for Each Set and Calculate Overall Performance
```bash
cd utils
./gen_score.sh
```


## 🏆 Leaderboard





<div align="center">
  <table style="margin: 0 auto; text-align: center;">
    <thead>
      <tr>
         <th class="tg-c3ow" colspan="14"></th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: none;">
        <td rowspan="2">Model</td>
        <td>Knowledge</td>
        <td>Reasoning</td>
        <td>Creativity</td>
        <td>Fluency</td>
        <td>Clarity</td>
        <td>Single-round</td>
        <td>Multi-round</td>
        <td>Instruction Following</td>
        <td>Emotion-Aware</td>
        <td>Safety Alignment</td>
        <td>Latency</td>
        <td>Robustness</td>
        <td rowspan="2">Overall</td>
      </tr>
      <tr>
        <td>Acc(%)</td>
        <td>Score</td>
        <td>Score</td>
        <td>UTMOS</td>
        <td>WER(%)</td>
        <td>Score</td>
        <td>Score</td>
        <td>FR(%)</td>
        <td>Score</td>
        <td>RR(%)</td>
        <td>RTF</td>
        <td>Avg PR (%)</td>
      </tr>
      <tr>
        <td colspan="14">Tiny Models</td>
      </tr>
      <tr>
        <td>Mini-Omni</td>
        <td>2.20</td>
        <td>1.291</td>
        <td>1.4725</td>
        <td>4.435</td>
        <td>19.571</td>
        <td>1.645</td>
        <td> - </td>
        <td>0</td>
        <td>5.428</td>
        <td>81.25</td>
        <td>0.3781</td>
        <td>84.14</td>
        <td>40.646</td>
      </tr>
      <tr>
        <td>Mini-Omni2</td>
        <td>4.65</td>
        <td>1.501</td>
        <td>1.8025</td>
        <td>4.413</td>
        <td>36.269</td>
        <td>1.915</td>
        <td> - </td>
        <td>0.11</td>
        <td>5.709</td>
        <td>88.50</td>
        <td>0.2001</td>
        <td>82.26</td>
        <td>43.224</td>
      </tr>
      <tr>
        <td>SLAM-Omni</td>
        <td>12.05</td>
        <td>1.875</td>
        <td>2.5175</td>
        <td>4.424</td>
        <td>6.065</td>
        <td>2.880</td>
        <td>1.9800</td>
        <td>3.11</td>
        <td>6.452</td>
        <td><b>90.25<br></td>
        <td>0.4925</td>
        <td>77.91</td>
        <td>54.649</td>
      </tr>
      <tr>
        <td>VocalNet-1B</td>
        <td><b>43.00<br></td>
        <td><b>2.869<br></td>
        <td><b>3.1800<br></td>
        <td><b>4.437<br></td>
        <td><b>5.123<br></td>
        <td><b>3.335<br></td>
        <td><b>3.2550<br></td>
        <td><b>16.11<br></td>
        <td><b>6.754<br></td>
        <td>89.00</td>
        <td><b>0.1632<br></td>
        <td><b>92.42<br></td>
        <td><b>66.632<br></td>
      </tr>
      <tr>
        <td colspan="14">Base Models</td>
      </tr>
      <tr>
        <td>LLaMA-Omni</td>
        <td>37.40</td>
        <td>2.591</td>
        <td>2.8475</td>
        <td>3.959</td>
        <td>2.842</td>
        <td>3.300</td>
        <td>3.1525</td>
        <td>14.89</td>
        <td>6.128</td>
        <td>27.75</td>
        <td><b>0.0958<br></td>
        <td>83.59</td>
        <td>57.107</td>
      </tr>
      <tr>
        <td>Freeze-Omni</td>
        <td>44.25</td>
        <td>3.530</td>
        <td>2.8850</td>
        <td>4.381</td>
        <td>11.460</td>
        <td>2.960</td>
        <td>-</td>
        <td>12.05</td>
        <td>6.164</td>
        <td>86.50</td>
        <td>0.2618</td>
        <td>65.25</td>
        <td>58.362</td>
      </tr>
      <tr>
        <td>Baichuan-Omni-1.5</td>
        <td>49.85</td>
        <td>3.770</td>
        <td><b>3.5900<br></td>
        <td>4.014</td>
        <td>23.452</td>
        <td><b>3.840<br></td>
        <td>-</td>
        <td>28.89</td>
        <td>5.424</td>
        <td>83.00</td>
        <td>1.4900</td>
        <td>74.85</td>
        <td>60.239</td>
      </tr>
      <tr>
        <td>GLM-4-Voice</td>
        <td>56.40</td>
        <td>3.641</td>
        <td>3.2900</td>
        <td>3.869</td>
        <td>11.565</td>
        <td>3.615</td>
        <td>3.7300</td>
        <td>31.67</td>
        <td>6.904</td>
        <td>71.50</td>
        <td>0.7870</td>
        <td>57.10</td>
        <td>61.388</td>
      </tr>
      <tr>
        <td>Kimi-Audio</td>
        <td>62.15</td>
        <td>3.132</td>
        <td>3.0950</td>
        <td>2.360</td>
        <td>38.001</td>
        <td>3.150</td>
        <td>3.5350</td>
        <td><b>48.59<br></td>
        <td>6.838</td>
        <td>83.35</td>
        <td>0.7331</td>
        <td><b>93.20<br></td>
        <td>62.382</td>
      </tr>
      <tr>
        <td>MiniCPM-o 2.6</td>
        <td><b>70.00<br></td>
        <td>3.648</td>
        <td>3.3550</td>
        <td>4.054</td>
        <td>18.735</td>
        <td>3.165</td>
        <td>3.6675</td>
        <td>30.00</td>
        <td>7.080</td>
        <td>83.25</td>
        <td>0.4509</td>
        <td>87.27</td>
        <td>63.886</td>
      </tr>
      <tr>
        <td>VITA-Audio-Plus-Vanilla</td>
        <td>52.00</td>
        <td>4.183</td>
        <td>3.2800</td>
        <td>4.173</td>
        <td>4.858</td>
        <td>3.520</td>
        <td>-</td>
        <td>33.59</td>
        <td>6.843</td>
        <td>88.25</td>
        <td>0.4645</td>
        <td>89.53</td>
        <td>71.795</td>
      </tr>
      <tr>
        <td>Qwen2.5-Omni</td>
        <td>69.50</td>
        <td><b>4.361<br></td>
        <td>3.1825</td>
        <td>4.174</td>
        <td><b>1.154<br></td>
        <td>3.538</td>
        <td><b>4.0125<br></td>
        <td>27.00</td>
        <td>6.386</td>
        <td>71.75</td>
        <td>1.7243</td>
        <td>91.86</td>
        <td>73.327</td>
      </tr>
      <tr>
        <td>VocalNet-8B</td>
        <td>67.95</td>
        <td>3.748</td>
        <td>3.5050</td>
        <td><b>4.449<br></td>
        <td>4.686</td>
        <td>3.530</td>
        <td>3.9175</td>
        <td>35.89</td>
        <td><b>7.117<br></td>
        <td><b>92.25<br></td>
        <td>0.2496</td>
        <td>92.66</td>
        <td><b>74.639<br></td>
      </tr>
      <tr>
        <td colspan="14">Real Time API</td>
      </tr>
      <tr>
        <td>Qwen-Omni-Turbo</td>
        <td>64.95</td>
        <td>4.058</td>
        <td>3.1575</td>
        <td>4.405</td>
        <td>1.656</td>
        <td>3.420</td>
        <td>3.9775</td>
        <td>22.11</td>
        <td>6.226</td>
        <td>65.25</td>
        <td>-</td>
        <td>90.64</td>
        <td>70.729</td>
      </tr>  
      <tr>
        <td colspan="14">Cascade System</td>
      </tr>
      <tr>
        <td>Whisper+GPT-4o+CosyVoice</td>
        <td><b>86.20<br></td>
        <td>4.138</td>
        <td><b>3.7500<br></td>
        <td><b>4.474<br></td>
        <td>4.955</td>
        <td>3.625</td>
        <td><b>4.2050<br></td>
        <td><b>66.33<br></td>
        <td>6.769</td>
        <td>91.50</td>
        <td>-</td>
        <td>90.79</td>
        <td><b>80.291<br></td>
      </tr>  
    <thead>
      <tr>
         <th class="tg-c3ow" colspan="14"></th>
      </tr>
    </thead>
    </tbody>
  </table>
</div>


<br> 
<br> 


## 🌞 Acknowledgements

- Whisper: VocalBench uses Whisper for speech recognition.
- Emotion2vec: VocalBench uses emotion2vec_plus_large for emotion recognition.
- UTMOS: VocalBench uses UTMOS to quantify the acoustic quality.
- Qwen2.5-Max: VocalBench uses Qwen2.5-Max for LLM evaluation.

<br> 
<br> 

