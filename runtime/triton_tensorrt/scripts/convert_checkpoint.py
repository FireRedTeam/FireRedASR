import argparse
import json
import os
import time

import torch
from safetensors.torch import save_file

import tensorrt_llm
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType
from tensorrt_llm.models.convert_utils import weight_only_quantize_dict
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the FireRedASR model.pth.tar checkpoint.")
    parser.add_argument('--output_dir', type=str, default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--logits_dtype', type=str, default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    return parser.parse_args()


def get_decoder_config(model_args, dtype: str, logits_dtype: str, quant_algo: QuantAlgo) -> dict:
    return {
        'architecture': "DecoderModel",
        'dtype': dtype,
        'logits_dtype': logits_dtype,
        'num_hidden_layers': model_args.n_layers_dec,
        'num_attention_heads': model_args.n_head,
        'hidden_size': model_args.d_model,
        'norm_epsilon': 1e-5,
        'vocab_size': model_args.odim,
        'hidden_act': "gelu",
        'use_parallel_embedding': False,
        'embedding_sharding_dim': 0,
        'max_position_embeddings': model_args.pe_maxlen,
        'use_prompt_tuning': False,
        'head_size': model_args.d_model // model_args.n_head,
        'has_position_embedding': True,
        'layernorm_type': LayerNormType.LayerNorm,
        'has_attention_qkvo_bias': True,
        'has_mlp_bias': True,
        'has_model_final_layernorm': True,
        'has_embedding_layernorm': False,
        'has_embedding_scale': True, # FireRedASR scales the embedding
        'ffn_hidden_size': 4 * model_args.d_model,
        'q_scaling': 1.0,
        'layernorm_position': LayerNormPositionType.pre_layernorm,
        'relative_attention': False,
        'max_distance': 0,
        'num_buckets': 0,
        'model_type': 'whisper', # To align with Whisper decoder architecture in TRT-LLM
        'rescale_before_lm_head': False,
        'encoder_hidden_size': model_args.d_model,
        'encoder_num_heads': model_args.n_head,
        'encoder_head_size': None,
        'skip_cross_kv': False,
        'quantization': {
            'quant_algo': quant_algo
        },
    }

def remap_state_dict(original_state_dict):
    new_state_dict = {}
    for key, value in original_state_dict.items():
        if key.startswith("decoder."):
            new_key = key
            # Top-level decoder module renames
            new_key = new_key.replace("decoder.tgt_word_emb.", "decoder.token_embedding.")
            new_key = new_key.replace("decoder.layer_stack.", "decoder.blocks.")
            new_key = new_key.replace("decoder.layer_norm_out.", "decoder.ln.")
            new_key = new_key.replace("decoder.tgt_word_prj.", "decoder.output_projection.")

            # ResidualAttentionBlock internal layer renames
            new_key = new_key.replace(".self_attn_norm.", ".attn_ln.")
            new_key = new_key.replace(".self_attn.", ".attn.")
            new_key = new_key.replace(".cross_attn_norm.", ".cross_attn_ln.")
            new_key = new_key.replace(".cross_attn.", ".cross_attn.")
            new_key = new_key.replace(".mlp_norm.", ".mlp_ln.")

            # Inlined PositionwiseFeedForward renames
            new_key = new_key.replace(".mlp.w_1.", ".mlp.0.")
            new_key = new_key.replace(".mlp.w_2.", ".mlp.2.")

            # MultiHeadAttention submodule renames
            new_key = new_key.replace(".w_qs.", ".query.")
            new_key = new_key.replace(".w_ks.", ".key.")
            new_key = new_key.replace(".w_vs.", ".value.")
            new_key = new_key.replace(".fc.", ".out.")

            new_state_dict[new_key] = value
    
    # Manually handle sinusoidal positional encoding -> learnable embedding
    if "decoder.positional_encoding.pe" in original_state_dict:
         new_state_dict["decoder.positional_embedding"] = original_state_dict["decoder.positional_encoding.pe"].squeeze(0)

    return new_state_dict


def convert_firered_decoder(model_args, model_params, quant_algo: str = None):
    weights = {}
    
    # The original model shares embedding and projection weights.
    # TRT-LLM's DecoderModel expects separate lm_head.weight
    weights['transformer.vocab_embedding.weight'] = model_params['decoder.token_embedding.weight']
    weights['lm_head.weight'] = model_params['decoder.output_projection.weight']
    weights['transformer.position_embedding.weight'] = model_params['decoder.positional_embedding']

    for i in range(model_args.n_layers_dec):
        trtllm_layer_name_prefix = f'transformer.layers.{i}'

        # Self Attention
        q_w = model_params[f'decoder.blocks.{i}.attn.query.weight']
        k_w = model_params[f'decoder.blocks.{i}.attn.key.weight']
        v_w = model_params[f'decoder.blocks.{i}.attn.value.weight']
        weights[f'{trtllm_layer_name_prefix}.self_attention.qkv.weight'] = torch.cat([q_w, k_w, v_w], dim=0)
        
        q_b = model_params[f'decoder.blocks.{i}.attn.query.bias']
        # The key projection has no bias in Whisper's MultiHeadAttention
        k_b = torch.zeros_like(q_b)
        v_b = model_params[f'decoder.blocks.{i}.attn.value.bias']
        weights[f'{trtllm_layer_name_prefix}.self_attention.qkv.bias'] = torch.cat([q_b, k_b, v_b], dim=0)
        
        weights[f'{trtllm_layer_name_prefix}.self_attention.dense.weight'] = model_params[f'decoder.blocks.{i}.attn.out.weight']
        weights[f'{trtllm_layer_name_prefix}.self_attention.dense.bias'] = model_params[f'decoder.blocks.{i}.attn.out.bias']
        weights[f'{trtllm_layer_name_prefix}.self_attention_layernorm.weight'] = model_params[f'decoder.blocks.{i}.attn_ln.weight']
        weights[f'{trtllm_layer_name_prefix}.self_attention_layernorm.bias'] = model_params[f'decoder.blocks.{i}.attn_ln.bias']

        # Cross Attention
        q_w = model_params[f'decoder.blocks.{i}.cross_attn.query.weight']
        k_w = model_params[f'decoder.blocks.{i}.cross_attn.key.weight']
        v_w = model_params[f'decoder.blocks.{i}.cross_attn.value.weight']
        weights[f'{trtllm_layer_name_prefix}.cross_attention.qkv.weight'] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = model_params[f'decoder.blocks.{i}.cross_attn.query.bias']
        # The key projection has no bias in Whisper's MultiHeadAttention
        k_b = torch.zeros_like(q_b)
        v_b = model_params[f'decoder.blocks.{i}.cross_attn.value.bias']
        weights[f'{trtllm_layer_name_prefix}.cross_attention.qkv.bias'] = torch.cat([q_b, k_b, v_b], dim=0)

        weights[f'{trtllm_layer_name_prefix}.cross_attention.dense.weight'] = model_params[f'decoder.blocks.{i}.cross_attn.out.weight']
        weights[f'{trtllm_layer_name_prefix}.cross_attention.dense.bias'] = model_params[f'decoder.blocks.{i}.cross_attn.out.bias']
        weights[f'{trtllm_layer_name_prefix}.cross_attention_layernorm.weight'] = model_params[f'decoder.blocks.{i}.cross_attn_ln.weight']
        weights[f'{trtllm_layer_name_prefix}.cross_attention_layernorm.bias'] = model_params[f'decoder.blocks.{i}.cross_attn_ln.bias']

        # MLP
        weights[f'{trtllm_layer_name_prefix}.mlp.fc.weight'] = model_params[f'decoder.blocks.{i}.mlp.0.weight']
        weights[f'{trtllm_layer_name_prefix}.mlp.fc.bias'] = model_params[f'decoder.blocks.{i}.mlp.0.bias']
        weights[f'{trtllm_layer_name_prefix}.mlp.proj.weight'] = model_params[f'decoder.blocks.{i}.mlp.2.weight']
        weights[f'{trtllm_layer_name_prefix}.mlp.proj.bias'] = model_params[f'decoder.blocks.{i}.mlp.2.bias']
        weights[f'{trtllm_layer_name_prefix}.mlp_layernorm.weight'] = model_params[f'decoder.blocks.{i}.mlp_ln.weight']
        weights[f'{trtllm_layer_name_prefix}.mlp_layernorm.bias'] = model_params[f'decoder.blocks.{i}.mlp_ln.bias']

    weights['transformer.ln_f.weight'] = model_params['decoder.ln.weight']
    weights['transformer.ln_f.bias'] = model_params['decoder.ln.bias']

    if quant_algo is not None:
        return weight_only_quantize_dict(weights, quant_algo=quant_algo)
    return weights


if __name__ == '__main__':
    print(f"Using TensorRT-LLM version: {tensorrt_llm.__version__}")
    args = parse_arguments()
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    quant_algo = None
    if args.use_weight_only and args.weight_only_precision == 'int8':
        quant_algo = QuantAlgo.W8A16
    elif args.use_weight_only and args.weight_only_precision == 'int4':
        quant_algo = QuantAlgo.W4A16

    # Load the original checkpoint
    package = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model_args = package["args"]
    original_state_dict = package["model_state_dict"]
    print(f"Successfully loaded checkpoint from {args.model_path}")
    print("Original model args:", model_args)

    # Remap state dict keys for Whisper compatibility
    remapped_state_dict = remap_state_dict(original_state_dict)
    
    # Set tensor dtype
    tensor_dtype = getattr(torch, args.dtype)
    for key, value in remapped_state_dict.items():
        remapped_state_dict[key] = value.to(tensor_dtype)

    # Generate config and convert weights
    print("Converting decoder checkpoint...")
    decoder_config = get_decoder_config(model_args, args.dtype, args.logits_dtype, quant_algo)
    decoder_weights = convert_firered_decoder(model_args, remapped_state_dict, quant_algo)
    
    # Save the decoder config and weights
    decoder_save_dir = os.path.join(args.output_dir, "decoder")
    if not os.path.exists(decoder_save_dir):
        os.makedirs(decoder_save_dir)

    with open(os.path.join(decoder_save_dir, 'config.json'), 'w') as f:
        json.dump(decoder_config, f, indent=4)
        
    save_file(decoder_weights, os.path.join(decoder_save_dir, f'rank0.safetensors'))

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Checkpoint successfully converted and saved to {args.output_dir}.')
    print(f'Total time of converting checkpoints: {t}')
