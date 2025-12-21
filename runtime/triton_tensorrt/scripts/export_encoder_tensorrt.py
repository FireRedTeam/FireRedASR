#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Zengwei Yao)
# Copyright         2025  Nvidia Corp.        (authors: Yuekai Zhang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script exports a pre-trained FireRedASR encoder model from PyTorch to
ONNX and TensorRT.

Usage:

python3 examples/export_encoder_tensorrt.py \
    --model-dir /path/to/your/model_dir \
    --tensorrt-model-dir ./tensorrt_models \
    --trt-engine-file-name encoder.plan
"""

import argparse
import logging
from pathlib import Path

import torch
import tensorrt as trt


def get_parser() -> argparse.ArgumentParser:
    """Get the command-line argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="The model directory that contains model checkpoint.",
    )

    parser.add_argument(
        "--onnx-model-path",
        type=str,
        default=None,
        help="If specified, we will directly use this onnx model to generate "
        "the tensorrt engine",
    )

    parser.add_argument(
        "--idim",
        type=int,
        default=80,
        help="The input dimension of the model. This is required when "
        "--onnx-model-path is specified.",
    )

    parser.add_argument(
        "--tensorrt-model-dir",
        type=str,
        default="exp",
        help="Directory to save the exported models.",
    )

    parser.add_argument(
        "--trt-engine-file-name",
        type=str,
        default="encoder.plan",
        help="The name of the TensorRT engine file.",
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version.",
    )

    return parser


def export_encoder_onnx(
    encoder: torch.nn.Module,
    filename: str,
    idim: int,
    opset_version: int = 17,
) -> None:
    """Export the conformer encoder model to ONNX format."""
    logging.info("Exporting encoder to ONNX")
    encoder.half()

    # Create dummy inputs
    seq_len = 400  # A typical sequence length
    batch_size = 1
    padded_input = torch.randn(batch_size, seq_len, idim, dtype=torch.float16)
    input_lengths = torch.tensor([seq_len] * batch_size, dtype=torch.int32)

    # Export
    torch.onnx.export(
        encoder,
        (padded_input, input_lengths),
        filename,
        opset_version=opset_version,
        input_names=["padded_input", "input_lengths"],
        output_names=["enc_output", "output_lengths", "src_mask"],
        dynamic_axes={
            "padded_input": {0: "batch_size", 1: "seq_len"},
            "input_lengths": {0: "batch_size"},
            "enc_output": {0: "batch_size", 1: "seq_len_out"},
            "output_lengths": {0: "batch_size",},
            "src_mask": {0: "batch_size", 2: "seq_len_out"},
        },
    )
    logging.info(f"Exported encoder to {filename}")


def get_trt_kwargs_dynamic_batch(
    idim: int,
    min_batch_size: int = 1,
    opt_batch_size: int = 4,
    max_batch_size: int = 64,
):
    """Get keyword arguments for TensorRT with dynamic batch size."""
    min_seq_len = 50
    opt_seq_len = 400
    max_seq_len = 3000

    min_shape = [(min_batch_size, min_seq_len, idim), (min_batch_size,)]
    opt_shape = [(opt_batch_size, opt_seq_len, idim), (opt_batch_size,)]
    max_shape = [(max_batch_size, max_seq_len, idim), (max_batch_size,)]
    input_names = ["padded_input", "input_lengths"]
    return {
        "min_shape": min_shape,
        "opt_shape": opt_shape,
        "max_shape": max_shape,
        "input_names": input_names,
    }


def convert_onnx_to_trt(
    trt_model: str, trt_kwargs: dict, onnx_model: str, dtype: torch.dtype = torch.float16
) -> None:
    """Convert an ONNX model to a TensorRT engine."""
    logging.info("Converting ONNX to TensorRT engine...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    if dtype == torch.float16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()

    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError(f'Failed to parse {onnx_model}')

    for i, name in enumerate(trt_kwargs['input_names']):
        profile.set_shape(
            name,
            trt_kwargs['min_shape'][i],
            trt_kwargs['opt_shape'][i],
            trt_kwargs['max_shape'][i]
        )
    
    config.add_optimization_profile(profile)
    
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except Exception as e:
        logging.error(f"TensorRT engine build failed: {e}")
        return

    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Successfully converted ONNX to TensorRT.")


@torch.no_grad()
def main():
    """Main function to export the model."""
    parser = get_parser()
    args = parser.parse_args()

    tensorrt_model_dir = Path(args.tensorrt_model_dir)
    tensorrt_model_dir.mkdir(parents=True, exist_ok=True)

    if args.onnx_model_path:
        logging.info(f"Using provided ONNX model: {args.onnx_model_path}")
        if not args.idim:
            raise ValueError("--idim is required when using --onnx-model-path")
        idim = args.idim
        encoder_onnx_file = Path(args.onnx_model_path)
        if not encoder_onnx_file.is_file():
            raise FileNotFoundError(f"ONNX model not found at {encoder_onnx_file}")
    else:
        from fireredasr.models.fireredasr import load_fireredasr_aed_model
        if not args.model_dir:
            raise ValueError(
                "--model-dir is required if --onnx-model-path is not provided"
            )

        logging.info("Exporting ONNX model from PyTorch checkpoint")
        model_dir = Path(args.model_dir)
        model_path = model_dir / "model.pth.tar"

        # Load model to get encoder
        package = torch.load(model_path, map_location="cpu", weights_only=False)
        model_args = package["args"]
        idim = model_args.idim
        # We have to load the full AED model to get the encoder with weights
        model = load_fireredasr_aed_model(str(model_path))
        encoder = model.encoder
        encoder.eval()

        # Export ONNX
        encoder_onnx_file = tensorrt_model_dir / "encoder.fp16.onnx"
        export_encoder_onnx(
            encoder=encoder,
            filename=str(encoder_onnx_file),
            idim=idim,
            opset_version=args.opset_version,
        )

    # Convert ONNX to TensorRT
    trt_engine_file = tensorrt_model_dir / args.trt_engine_file_name
    trt_kwargs = get_trt_kwargs_dynamic_batch(idim=idim)
    convert_onnx_to_trt(
        trt_model=str(trt_engine_file),
        trt_kwargs=trt_kwargs,
        onnx_model=str(encoder_onnx_file),
        dtype=torch.float16,
    )

    logging.info("Done!")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
