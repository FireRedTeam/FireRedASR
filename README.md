<div align="center">
<h1>FireRedASR: Open-Source Industrial-Grade
<br>
Automatic Speech Recognition Models</h1>

</div>

[[Paper]](https://arxiv.org/pdf/2501.14350)
[[Model]](https://huggingface.co/fireredteam)
[[Blog]](https://fireredteam.github.io/demos/firered_asr/)
[[Demo]](https://huggingface.co/spaces/FireRedTeam/FireRedASR)

FireRedASR is a family of open-source industrial-grade automatic speech recognition (ASR) models supporting Mandarin, Chinese dialects and English, achieving a new state-of-the-art (SOTA) on public Mandarin ASR benchmarks, while also offering outstanding singing lyrics recognition capability.


## 🔥 News
- [2025/02/17] We release [FireRedASR-LLM-L](https://huggingface.co/fireredteam/FireRedASR-LLM-L/tree/main) model weights.
- [2025/01/24] We release [technical report](https://arxiv.org/pdf/2501.14350), [blog](https://fireredteam.github.io/demos/firered_asr/), and [FireRedASR-AED-L](https://huggingface.co/fireredteam/FireRedASR-AED-L/tree/main) model weights.


## Method

FireRedASR is designed to meet diverse requirements in superior performance and optimal efficiency across various applications. It comprises two variants:
- FireRedASR-LLM: Designed to achieve state-of-the-art (SOTA) performance and to enable seamless end-to-end speech interaction. It adopts an Encoder-Adapter-LLM framework leveraging large language model (LLM) capabilities.
- FireRedASR-AED: Designed to balance high performance and computational efficiency and to serve as an effective speech representation module in LLM-based speech models. It utilizes an Attention-based Encoder-Decoder (AED) architecture.

![Model](/assets/FireRedASR_model.png)


## Evaluation
Results are reported in Character Error Rate (CER%) for Chinese and Word Error Rate (WER%) for English.

### Evaluation on Public Mandarin ASR Benchmarks
| Model            | #Params | aishell1 | aishell2 | ws\_net  | ws\_meeting | Average-4 |
|:----------------:|:-------:|:--------:|:--------:|:--------:|:-----------:|:---------:|
| FireRedASR-LLM   | 8.3B | 0.76 | 2.15 | 4.60 | 4.67 | 3.05 |
| FireRedASR-AED   | 1.1B | 0.55 | 2.52 | 4.88 | 4.76 | 3.18 |
| Seed-ASR         | 12B+ | 0.68 | 2.27 | 4.66 | 5.69 | 3.33 |
| Qwen-Audio       | 8.4B | 1.30 | 3.10 | 9.50 | 10.87 | 6.19 |
| SenseVoice-L     | 1.6B | 2.09 | 3.04 | 6.01 | 6.73 | 4.47 |
| Whisper-Large-v3 | 1.6B | 5.14 | 4.96 | 10.48 | 18.87 | 9.86 |
| Paraformer-Large | 0.2B | 1.68 | 2.85 | 6.74 | 6.97 | 4.56 |

`ws` means WenetSpeech.

### Evaluation on Public Chinese Dialect and English ASR Benchmarks
|Test Set       | KeSpeech | LibriSpeech test-clean | LibriSpeech test-other  |
| :------------:| :------: | :--------------------: | :----------------------:|
|FireRedASR-LLM | 3.56 | 1.73 | 3.67 |
|FireRedASR-AED | 4.48 | 1.93 | 4.44 |
|Previous SOTA Results | 6.70 | 1.82 | 3.50 |


## Usage
Download model files from [huggingface](https://huggingface.co/fireredteam) and place them in the folder `pretrained_models`.

If you want to use `FireRedASR-LLM-L`, you also need to download [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) and place it in the folder `pretrained_models`. Then, go to folder `FireRedASR-LLM-L` and run `$ ln -s ../Qwen2-7B-Instruct`


### Setup
Create a Python environment and install dependencies
```bash
$ git clone https://github.com/FireRedTeam/FireRedASR.git
$ conda create --name fireredasr python=3.10
$ conda activate fireredasr
$ pip install -r requirements.txt
```

Set up Linux PATH and PYTHONPATH
```
$ export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
$ export PYTHONPATH=$PWD/:$PYTHONPATH
```

Convert audio to 16kHz 16-bit PCM format
```
ffmpeg -i input_audio -ar 16000 -ac 1 -acodec pcm_s16le -f wav output.wav
```

### Quick Start
```bash
$ cd examples
$ bash inference_fireredasr_aed.sh
$ bash inference_fireredasr_llm.sh
```

### Command-line Usage
```bash
$ speech2text.py --help
$ speech2text.py --wav_path examples/wav/BAC009S0764W0121.wav --asr_type "aed" --model_dir pretrained_models/FireRedASR-AED-L
$ speech2text.py --wav_path examples/wav/BAC009S0764W0121.wav --asr_type "llm" --model_dir pretrained_models/FireRedASR-LLM-L
```

### Python Usage
```python
from fireredasr.models.fireredasr import FireRedAsr

batch_uttid = ["BAC009S0764W0121"]
batch_wav_path = ["examples/wav/BAC009S0764W0121.wav"]

# FireRedASR-AED
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L")
results = model.transcribe(
    batch_uttid,
    batch_wav_path,
    {
        "use_gpu": 1,
        "beam_size": 3,
        "nbest": 1,
        "decode_max_len": 0,
        "softmax_smoothing": 1.25,
        "aed_length_penalty": 0.6,
        "eos_penalty": 1.0
    }
)
print(results)


# FireRedASR-LLM
model = FireRedAsr.from_pretrained("llm", "pretrained_models/FireRedASR-LLM-L")
results = model.transcribe(
    batch_uttid,
    batch_wav_path,
    {
        "use_gpu": 1,
        "beam_size": 3,
        "decode_max_len": 0,
        "decode_min_len": 0,
        "repetition_penalty": 3.0,
        "llm_length_penalty": 1.0,
        "temperature": 1.0
    }
)
print(results)
```

## Usage Tips
### Batch Beam Search
- When performing batch beam search with FireRedASR-LLM, please ensure that the input lengths of the utterances are similar. If there are significant differences in utterance lengths, shorter utterances may experience repetition issues. You can either sort your dataset by length or set `batch_size` to 1 to avoid the repetition issue.

### Input Length Limitations
- FireRedASR-AED supports audio input up to 60s. Input longer than 60s may cause hallucination issues, and input exceeding 200s will trigger positional encoding errors.
- FireRedASR-LLM supports audio input up to 30s. The behavior for longer input is currently unknown.


## Acknowledgements
Thanks to the following open-source works:
- [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [icefall/ASR_LLM](https://github.com/k2-fsa/icefall/tree/master/egs/speech_llm/ASR_LLM)
- [WeNet](https://github.com/wenet-e2e/wenet)
- [Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer)


## Citation
```bibtex
@article{xu2025fireredasr,
  title={FireRedASR: Open-Source Industrial-Grade Mandarin Speech Recognition Models from Encoder-Decoder to LLM Integration},
  author={Xu, Kai-Tuo and Xie, Feng-Long and Tang, Xu and Hu, Yao},
  journal={arXiv preprint arXiv:2501.14350},
  year={2025}
}
```
