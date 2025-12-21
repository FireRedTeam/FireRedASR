import json
import os
import sys
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from pathlib import Path

from .fireredasr_aed_tensorrt import FireRedAsrAedTensorRT
from .asr_feat import ASRFeatExtractor
from .aed_tokenizer import ChineseCharEnglishSpmTokenizer


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        
        self.engine_dir = self.model_config['parameters']['engine_dir']['string_value']
        self.checkpoint_dir = self.model_config['parameters']['checkpoint_dir']['string_value']
    
        self.device_id = int(args["model_instance_device_id"])
        self.device = torch.device(f"cuda:{self.device_id}")

        cmvn_path = os.path.join(self.checkpoint_dir, "cmvn.ark")
        self.feat_extractor = ASRFeatExtractor(cmvn_path, device_id=self.device_id)
        
        dict_path = os.path.join(self.checkpoint_dir, "dict.txt")
        spm_model = os.path.join(self.checkpoint_dir, "train_bpe1000.model")
        self.tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)

        self.model = FireRedAsrAedTensorRT.from_model_dir(self.engine_dir, self.tokenizer, device_id=self.device_id)

    def execute(self, requests):
        responses = []
        all_wavs = []
        request_sample_counts = []
        
        for request in requests:
            wav_tensor = pb_utils.get_input_tensor_by_name(request, "WAV")
            wav_lens_tensor = pb_utils.get_input_tensor_by_name(request, "WAV_LENS")
            
            wav_data = wav_tensor.as_numpy()

            wav_len_data = wav_lens_tensor.as_numpy()
            valid_len = int(wav_len_data.flatten()[0])

            wav_seq = wav_data[0,:valid_len]
            wav_seq = wav_seq * (1 << 15)
            
            all_wavs.append(wav_seq)
            request_sample_counts.append(1)

        feats_pad, lengths, durs = self.feat_extractor(all_wavs)
        
        hyps = self.model.transcribe(
            feats_pad, 
            lengths,
            beam_size=1,
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.0,
            length_penalty=0.0,
            eos_penalty=1.0
        )

        all_results = []
        for hyp_list in hyps:
            if not hyp_list:
                all_results.append("")
                continue
                
            hyp = hyp_list[0]
            hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
            text = self.tokenizer.detokenize(hyp_ids)
            all_results.append(text)

        result_idx = 0
        for count in request_sample_counts:
            req_texts = []
            for _ in range(count):
                req_texts.append(all_results[result_idx])
                result_idx += 1
            
            out_np = np.array(req_texts, dtype=object).reshape(-1, 1)
            
            out_tensor = pb_utils.Tensor("TRANSCRIPTS", out_np)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
            
        return responses

    def finalize(self):
        print('Cleaning up FireRedASR model...')
