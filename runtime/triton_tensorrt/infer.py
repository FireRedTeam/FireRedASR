import argparse
import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
import torchaudio
import datasets
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np

# Add model directory to sys.path to handle the '1' directory which is not a valid python package name
# Assuming this script is located at FireRedASR/runtime/triton_tensorrt/infer.py
current_dir = Path(__file__).resolve().parent
model_dir = current_dir / "model_repo_fireredasr_aed" / "fireredasr" / "1"
sys.path.append(str(model_dir))

try:
    from fireredasr_aed_tensorrt import FireRedAsrAedTensorRT
    from asr_feat import ASRFeatExtractor
    from aed_tokenizer import ChineseCharEnglishSpmTokenizer
except ImportError as e:
    print(f"Error importing model modules: {e}")
    print(f"Added path: {model_dir}")
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(description="FireRedASR Inference")
    parser.add_argument(
        "--engine_dir", 
        type=str, 
        default="/workspace_yuekai/asr/FireRedASR/examples/FireRedASR-AED-L-TensorRT",
        help="TensorRT engine directory"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="/workspace_yuekai/asr/FireRedASR/examples/pretrained_models/FireRedASR-AED-L",
        help="Checkpoint directory containing cmvn, dict, spm"
    )
    parser.add_argument(
        "--huggingface_dataset", 
        type=str, 
        required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "--subset_name", 
        type=str, 
        default=None, 
        help="Dataset subset name"
    )
    parser.add_argument(
        "--split_name", 
        type=str, 
        default="test", 
        help="Dataset split name"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="results.txt", 
        help="Output file path"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size (per-device)"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4, 
        help="Num workers for dataloader"
    )
    parser.add_argument(
        "--prefetch", 
        type=int, 
        default=5, 
        help="Prefetch factor for dataloader"
    )
    return parser.parse_args()

def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    print(f"Inference on multiple gpus, this gpu {local_rank}, rank {rank}, world_size {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank

def data_collator(batch):
    ids = []
    wavs = []
    target_sr = 16000
    
    for item in batch:
        # Handle 'id' or 'audio_id' or key that identifies the utterance
        utt_id = item.get("id") or item.get("segment_id") or str(item.get("key", "unknown"))
        ids.append(utt_id)
        
        audio_info = item["audio"]
        audio = audio_info["array"]
        sr = audio_info["sampling_rate"]
        
        # Ensure audio is float32 and correct sampling rate
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()
        
        audio_tensor = audio_tensor * (1 << 15)

        if sr != target_sr:
             resampler = torchaudio.transforms.Resample(sr, target_sr)
             audio_tensor = resampler(audio_tensor)
        
        wavs.append(audio_tensor)
             
    return ids, wavs

def main():
    args = get_args()
    
    world_size, local_rank, rank = init_distributed()
    device_id = local_rank
    
    # Load Model Components
    cmvn_path = os.path.join(args.checkpoint_dir, "cmvn.ark")
    feat_extractor = ASRFeatExtractor(cmvn_path, device_id=device_id)

    dict_path = os.path.join(args.checkpoint_dir, "dict.txt")
    spm_model = os.path.join(args.checkpoint_dir, "train_bpe1000.model")
    tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)

    model = FireRedAsrAedTensorRT.from_model_dir(args.engine_dir, tokenizer, device_id=device_id)

    # Load Dataset
    print(f"Loading dataset: {args.huggingface_dataset} split: {args.split_name}")
    dataset = datasets.load_dataset(
        args.huggingface_dataset,
        args.subset_name,
        split=args.split_name,
        trust_remote_code=True,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        collate_fn=data_collator,
        shuffle=False
    )

    # Process
    if rank == 0:
        progress_bar = tqdm(total=len(dataset), desc="Decoding", unit="utts")

    results = []

    for batch_ids, batch_wavs in dataloader:

        feats_pad, lengths, durs = feat_extractor(batch_wavs)
        
        # Inference
        hyps_list = model.transcribe(
            feats_pad, 
            lengths,
            beam_size=1,
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.0,
            length_penalty=0.0,
            eos_penalty=1.0
        )
        
        for i, hyp_list in enumerate(hyps_list):
            hyp = hyp_list[0]
            hyp_ids = [int(id) for id in hyp["yseq"].tolist()]
            text = tokenizer.detokenize(hyp_ids)
            results.append(f"{batch_ids[i]}\t{text}")

        if rank == 0:
            progress_bar.update(len(batch_ids) * world_size)

    if rank == 0:
        progress_bar.close()
        
    # Write results per rank
    # all gather the results
    if world_size > 1:
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)
        results = [item for sublist in all_results for item in sublist]

    if rank == 0:
        output_file = args.output_file
        os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for line in results:
                f.write(line + "\n")
        print(f"Saved merged results to {output_file}")
                
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
