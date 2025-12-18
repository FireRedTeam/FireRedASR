import argparse
import requests
import soundfile as sf
import json
import numpy as np
from tritonclient.utils import np_to_triton_dtype

def main():
    parser = argparse.ArgumentParser(description="FireRedASR HTTP Client")
    parser.add_argument("--wav_path", type=str, default="../../examples/wav/TEST_MEETING_T0000000001_S00000.wav", help="Path to the wav file")
    args = parser.parse_args()

    url = "http://localhost:8000/v2/models/fireredasr/infer"
    wav_path = args.wav_path
    
    waveform, sr = sf.read(wav_path)
    assert sr == 16000, "Sample rate must be 16000"

    samples = np.array([waveform], dtype=np.float32)
    lengths = np.array([[len(waveform)]], dtype=np.int32)

    data = {
        "inputs": [
            {
                "name": "WAV",
                "shape": samples.shape,
                "datatype": "FP32",
                "data": samples.tolist()
            },
            {
                "name": "WAV_LENS",
                "shape": lengths.shape,
                "datatype": "INT32",
                "data": lengths.tolist(),
            }
        ]
    }
    rsp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=data,
        verify=False,
        params={"request_id": '0'}
    )
    result = rsp.json()
    transcripts = result["outputs"][0]["data"][0]
    print(transcripts)

if __name__ == "__main__":
    main()
