stage=$1
stop_stage=$2

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Downloading FireRedASR-AED-L-TensorRT and FireRedASR-AED-L"
    # huggingface-cli login
    huggingface-cli download yuekai/FireRedASR-AED-L-TensorRT --local-dir FireRedASR-AED-L-TensorRT
    huggingface-cli download FireRedTeam/FireRedASR-AED-L --local-dir FireRedASR-AED-L
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "export TensorRT engines"
    # assert engine_dir=./FireRedASR-AED-L-TensorRT
    bash scripts/export_tensorrt.sh
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Starting Triton server"
    export CUDA_VISIBLE_DEVICES="0"
    tritonserver --model-repository=model_repo_fireredasr_aed
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "http client"
    python3 http_client.py --wav_path ../../examples/wav/TEST_MEETING_T0000000001_S00000.wav
fi


if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Running Triton client"
    if [ ! -d "Triton-ASR-Client" ]; then
        git clone https://github.com/yuekaizhang/Triton-ASR-Client.git
    fi
    num_task=128
    dataset_name=yuekai/aishell
    subset_name=test
    split_name=test

    python3 Triton-ASR-Client/client.py \
        --server-addr localhost \
        --model-name fireredasr \
        --num-tasks $num_task \
        --log-dir ./log_fireredasr_$num_task \
        --huggingface_dataset $dataset_name \
        --subset_name $subset_name \
        --split_name $split_name \
        --compute-cer
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Running offline inference"
    export DISABLE_TORCH_DEVICE_SET=True
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    dataset_name=yuekai/aishell
    subset_name=test
    split_name=test

    torchrun --nproc_per_node=1 \
        infer.py \
        --engine_dir ./FireRedASR-AED-L-TensorRT \
        --checkpoint_dir ./FireRedASR-AED-L \
        --huggingface_dataset $dataset_name \
        --subset_name $subset_name \
        --split_name $split_name \
        --batch_size 64

fi