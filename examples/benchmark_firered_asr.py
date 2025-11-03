import os
import time
import torch
import numpy as np
from tqdm import tqdm

import librosa
import soundfile as sf
import argparse
import torch

from fireredasr.models.fireredasr import FireRedAsr

from torch.profiler import profile as torch_profiler
from torch.profiler import ProfilerActivity, record_function


ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "XFORMERS") # Option: "NATIVE", "SDPA", "XFORMERS"


def load_model(model_path="pretrained_models/FireRedASR-AED-L"):
    print("==========Load model:========")
    model = FireRedAsr.from_pretrained("aed", model_path)
    model.model.half()
    model.model.cuda()
    model.model.eval()

    return model

def load_audio(wav_path):
    print("==========load audio:=========")
    audio, sr = sf.read(wav_path,dtype=np.float32)
    print(len(audio), audio.dtype)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio

def benchmark(model, wav_path, batch, warmpup=2, trials=10, enable_profile=False):
    batch_wav_path = [wav_path] * batch
    batch_uttid = list(range(batch))
    results = None
    total_dur = None
 
    preprocess_start = time.time()
    feats, lengths, durs = model.feat_extractor(batch_wav_path)
    feats = feats.half()
    feats, lengths = feats.cuda(), lengths.cuda()
    preprocess_dur = time.time() - preprocess_start
    print(f"preprocess_dur: {preprocess_dur:.3f} s")
    total_dur = sum(durs)
    print(f"total input duration: {total_dur:.3f} s")

    # Warmup
    print("==========warmup========")
    for _ in range(warmpup):
        with torch.no_grad():
            _ = model.model.transcribe(feats, lengths)

    # Benchmark
    print("==========start benchmark========")
    total_time = 0
    results = []
    rtf_list = []
    if enable_profile: 
        warmup=1
        trials=1
    for _ in tqdm(range(trials)):
        start = time.time()
        with torch.no_grad():
            if enable_profile:
                with torch_profiler(activities=[
                    ProfilerActivity.CPU, 
                    ProfilerActivity.CUDA], 
                    record_shapes=True, 
                    with_stack=True,
                    profile_memory=False) as prof:
                    #with record_function("model.model.transcribe"):
                    hyps = model.model.transcribe(feats, lengths)
                print(prof.key_averages().table(sort_by="cuda_time_total"))
                prof.export_chrome_trace(f"firered_asr_profile_{batch}_{ATTENTION_BACKEND}.json")
            else:
                hyps = model.model.transcribe(feats, lengths)
        total_time += time.time() - start
        elapsed = time.time() - start

        rtf = elapsed / total_dur if total_dur > 0 else 0
        for uttid, wav, hyp in zip(batch_uttid, batch_wav_path, hyps):
            hyp = hyp[0]  # only return 1-best
            hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
            text = model.tokenizer.detokenize(hyp_ids)
            results.append({"uttid": uttid, "text": text, "wav": wav,
                "rtf": f"{rtf:.4f}"})
        rtf_list.append(rtf)

    avg_latency = total_time / trials
    rps = batch / avg_latency
    for res in results:
        print(res)
    avg_rtf = sum(rtf_list) / len(rtf_list)
    return rps, avg_latency, avg_rtf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Benchmark scripts for FireRedASR', usage='%(prog)s [options]')
    parser.add_argument('-b', '--batch_sizes', type=int, nargs='+', default=1, help='List of batch sizes for performance evaluation')
    parser.add_argument('-m', '--model_path', type=str, default="pretrained_models/FireRedASR-AED-L", help='Path to model directory')
    parser.add_argument('-a', '--audio_path', type=str, default='examples/wav/TEST_MEETING_T0000000001_S00000.wav', help="Input audio path")
    parser.add_argument('-d', '--device', type=str, default='cuda', help="Target inference device")
    parser.add_argument('-p', '--profile', action='store_true', help='Enable torch profiler')
    args = parser.parse_args()
    audio_path = args.audio_path
    model_path = args.model_path
    device = args.device
    enable_profile = args.profile
    batch_sizes = args.batch_sizes # [1, 4, 8, 16, 32]
    model = load_model(model_path)
    
    if enable_profile:
        rps, avg_rtf, avg_latency = benchmark(model, audio_path, batch=1, enable_profile=True)
    else:            
        for batch in batch_sizes:
            print(f"=============== batch size {batch} ==========================")
            rps, avg_latency, avg_rtf = benchmark(model, audio_path, batch=batch)
            print(f"batch size: {batch}, average latency: {avg_latency:.3f}s | RPS: {rps:.2f}, avg RTF: {avg_rtf:.3f}")
