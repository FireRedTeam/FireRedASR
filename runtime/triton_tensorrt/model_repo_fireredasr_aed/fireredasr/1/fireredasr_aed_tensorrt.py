import torch
import json
from pathlib import Path
import os

import tensorrt_llm
from tensorrt_llm.runtime import ModelConfig, SamplingConfig, GenerationSession, Session, TensorInfo
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_torch
from collections import OrderedDict

def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config

class TrtEncoder:
    def __init__(self, engine_path, device_id=0, dtype='float16'):
        self.device_id = device_id
        self.dtype = dtype
        self.session = self.get_session(engine_path)

    def get_session(self, engine_path):
        torch.cuda.set_device(self.device_id)
        with open(engine_path, 'rb') as f:
            engine_buffer = f.read()
        session = Session.from_serialized_engine(engine_buffer)
        return session

    def forward(self, padded_input, input_lengths):
        torch.cuda.set_device(self.device_id)
        inputs = OrderedDict()
        inputs['padded_input'] = padded_input
        inputs['input_lengths'] = input_lengths

        input_info = [
            TensorInfo('padded_input', str_dtype_to_trt(self.dtype), padded_input.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'), input_lengths.shape),
        ]

        output_info = self.session.infer_shapes(input_info)
        
        outputs = {
            t.name: torch.empty(
                tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device=f'cuda:{self.device_id}'
            )
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        assert ok, "TrtEncoder execution failed"
        stream.synchronize()

        return outputs['enc_output'], outputs['output_lengths'], outputs['src_mask']


def remove_tensor_padding(input_tensor,
                          input_tensor_lengths=None,
                          pad_value=None):
    if pad_value:
        assert input_tensor_lengths is None, "input_tensor_lengths should be None when pad_value is provided"
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] !=
            pad_value), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    else:
        # Audio tensor case: batch, seq_len, feature_len
        # position_ids case: batch, seq_len
        assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(input_tensor.shape[0]):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)
    return output_tensor

class TrtLlmDecoder:

    def __init__(self, engine_dir, sos_id, eos_id, pad_id, runtime_mapping, device_id=0, debug_mode=False):
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.device_id = device_id

        self.decoder_config = read_config('decoder', engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode)

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        torch.cuda.set_device(self.device_id)
        serialize_path = engine_dir / 'decoder' / 'rank0.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config['max_batch_size'],
            max_beam_width=self.decoder_config['max_beam_width'],
            num_heads=self.decoder_config['num_attention_heads'],
            num_kv_heads=self.decoder_config['num_attention_heads'],
            hidden_size=self.decoder_config['hidden_size'],
            vocab_size=self.decoder_config['vocab_size'],
            cross_attention=True,
            num_layers=self.decoder_config['num_hidden_layers'],
            gpt_attention_plugin=self.decoder_config['plugin_config']
            ['gpt_attention_plugin'],
            remove_input_padding=self.decoder_config['plugin_config']
            ['remove_input_padding'],
            kv_cache_type=KVCacheType.PAGED
            if self.decoder_config['plugin_config']['paged_kv_cache'] == True
            else KVCacheType.CONTINUOUS,
            has_position_embedding=self.
            decoder_config['has_position_embedding'],
            dtype=self.decoder_config['dtype'],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode)

        return decoder_generation_session

    def generate(self,
                 decoder_input_ids,
                 encoder_outputs,
                 encoder_max_input_length,
                 encoder_output_lengths,
                 max_new_tokens=40,
                 num_beams=1,
                 length_penalty=0.0):
        torch.cuda.set_device(self.device_id)
        batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor([
            decoder_input_ids.shape[-1]
            for _ in range(decoder_input_ids.shape[0])
        ],
                                             dtype=torch.int32,
                                             device=f'cuda:{self.device_id}')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones([
            batch_size, decoder_max_input_length + max_new_tokens,
            encoder_max_input_length
        ]).int().to(f'cuda:{self.device_id}')

        sampling_config = SamplingConfig(end_id=self.eos_id,
                                         pad_id=self.pad_id,
                                         num_beams=num_beams,
                                         length_penalty=length_penalty)
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_max_input_length)

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).to(f'cuda:{self.device_id}')
        if self.decoder_config['plugin_config']['remove_input_padding']:
            decoder_input_ids = remove_tensor_padding(
                decoder_input_ids, pad_value=self.pad_id)
            decoder_input_ids = decoder_input_ids.contiguous()
            if encoder_outputs.dim() == 3:
                encoder_outputs = remove_tensor_padding(encoder_outputs,
                                                        encoder_output_lengths)
                encoder_outputs = encoder_outputs.contiguous()
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_output_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        return output_ids

class FireRedAsrAedTensorRT(torch.nn.Module):
    @classmethod
    def from_model_dir(cls, tensorrt_model_dir, tokenizer, device_id=0):
        return cls(tensorrt_model_dir, tokenizer, device_id=device_id)

    def __init__(self, tensorrt_model_dir, tokenizer, device_id=0):
        super().__init__()
        self.device_id = device_id
        
        encoder_engine_path = os.path.join(tensorrt_model_dir, "encoder.plan")
        self.encoder = TrtEncoder(encoder_engine_path, device_id=device_id)

        self.decoder_engine_dir_name = "trt_engine_float16"
        decoder_engine_dir = Path(os.path.join(tensorrt_model_dir, self.decoder_engine_dir_name))

        sos_id = tokenizer.dict.word2id["<sos>"]
        eos_id = tokenizer.dict.word2id["<eos>"]
        pad_id = tokenizer.dict.word2id["<pad>"]
        
        runtime_mapping = tensorrt_llm.Mapping()
        self.decoder = TrtLlmDecoder(
            engine_dir=decoder_engine_dir,
            sos_id=sos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            runtime_mapping=runtime_mapping,
            device_id=device_id
        )


    def transcribe(self, padded_input, input_lengths,
                   beam_size=1, nbest=1, decode_max_len=0,
                   softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0):
        device = f'cuda:{self.device_id}'
        padded_input, input_lengths = padded_input.to(device), input_lengths.to(device)
        enc_outputs, enc_output_lengths, enc_mask = self.encoder.forward(padded_input.to(torch.float16).contiguous(), input_lengths.to(torch.int32).contiguous())
        if "float16" in self.decoder_engine_dir_name:
            enc_outputs = enc_outputs.to(torch.float16).contiguous()
        else:
            assert "float32" in self.decoder_engine_dir_name, "Decoder engine directory name must contain 'float32' or 'float16'"
            enc_outputs = enc_outputs.to(torch.float32).contiguous()

        batch_size = padded_input.size(0)
        device = padded_input.device

        decoder_input_ids = torch.full((batch_size, 1), self.decoder.sos_id, dtype=torch.int32).to(device)
        encoder_max_input_length = enc_outputs.size(1)
        encoder_output_lengths = enc_mask.sum(dim=-1).to(torch.int32).squeeze(-1)
        max_new_tokens = decode_max_len if decode_max_len > 0 else enc_outputs.size(1)

        output_ids = self.decoder.generate(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=enc_outputs,
            encoder_max_input_length=encoder_max_input_length,
            encoder_output_lengths=encoder_output_lengths,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            length_penalty=length_penalty
        )

        if nbest > beam_size:
            nbest = beam_size
        
        nbest_hyps = []
        for i in range(batch_size):
            n_hyps = []
            for j in range(nbest):
                token_ids = output_ids[i, j, decoder_input_ids.size(1):].cpu()
                eos_index = (token_ids == self.decoder.eos_id).nonzero(as_tuple=True)[0]
                if len(eos_index) > 0:
                    token_ids = token_ids[:eos_index[0]]
                n_hyps.append({"yseq": token_ids})
            nbest_hyps.append(n_hyps)

        return nbest_hyps
