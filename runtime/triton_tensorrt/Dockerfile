

FROM nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3
LABEL maintainer="zhangyuekai@foxmail.com"

RUN apt-get update && apt-get install -y cmake
RUN git clone https://github.com/pytorch/audio.git && cd audio && git checkout c670ad8 && PATH=/usr/local/cuda/bin:$PATH python3 setup.py develop
RUN git clone https://github.com/csukuangfj/kaldifeat && \
    cd kaldifeat && \
    sed -i 's/in running_cuda_version//g' get_version.py && \
    python3 setup.py install && \
    cd -
RUN pip install kaldiio kaldialign tritonclient[grpc]==2.31 soundfile numpy scipy librosa
WORKDIR /workspace