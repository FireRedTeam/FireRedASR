
## Accelerating FireredASR-AED with NVIDIA Triton Inference Server and TensorRT-LLM

Contributed by Yuekai Zhang (NVIDIA).

### Quick Start

Launch the service directly with Docker Compose:
```sh
HF_TOKEN="hf_your_token" docker compose up
```

### Build the Docker Image

To build the image from scratch:
```sh
docker build . -f Dockerfile -t soar97/triton-fireredasr:25.06
```

### Run a Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "fireredasr-aed-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-fireredasr:25.06
```

### Understanding `run.sh`

The `run.sh` script orchestrates the entire workflow through numbered stages.

You can run a subset of stages with:
```sh
bash run.sh <start_stage> <stop_stage>
```
- `<start_stage>`: The stage to start from (0-5).
- `<stop_stage>`: The stage to stop after (0-5).

**Stages:**

- **Stage 0**: Download Models
- **Stage 1**: Export TensorRT Engines
- **Stage 2**: Start Triton Server
- **Stage 3**: HTTP Client Test
- **Stage 4**: Benchmark with Triton Client
- **Stage 5**: Offline Inference Benchmark

### Export Models and Launch Server

Inside the Docker container, prepare the models and start the Triton server by running stages 0-2:
```sh
# This command runs stages 0, 1, and 2
bash run.sh 0 2
```

### Single-Utterance HTTP Client

Sends a single HTTP inference request (Stage 3):
```sh
bash run.sh 3 3
# Or directly:
python3 http_client.py --wav_path ../../examples/wav/TEST_MEETING_T0000000001_S00000.wav
```

### Benchmark with Triton Client

To benchmark the running Triton server (Stage 4):
```sh
bash run.sh 4 4
```
This will clone `Triton-ASR-Client` and run evaluation on the `yuekai/aishell` dataset.

### Offline Inference Benchmark

To run offline inference directly (Stage 5):
```sh
bash run.sh 5 5
```

### Benchmark Results

Decode aishell1 test 10 hours data on a single H20 GPU

| Method | Time (secs) |
| :--- | :--- |
| PyTorch Offline Inference | 760 |
| TensorRT Offline Inference | 60 |

### Acknowledgements

This work originates from the NVIDIA CISI project. For more multimodal resources, please see [mair-hub](https://github.com/nvidia-china-sae/mair-hub).
