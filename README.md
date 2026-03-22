mjpython -m reachy_mini.daemon.app.main --sim

source reachy_mini_env/bin/activate
reachy-mini-daemon


reachy-mini-daemon
https://huggingface.co/docs/reachy_mini/SDK/quickstart



---------------------------------------------

https://github.com/pollen-robotics/reachy-mini-desktop-app


---------------------------------------------
cd reachy-mini-desktop-app/


yarn tauri:dev


---------------------------------------------


check volume

curl -s http://127.0.0.1:8000/api/volume/current



--------------------------------------------


source .venv/bin/activate



VLM to speak

python3 VLM_speaker.py --no-device-map


--------------------------------------------

ASR (Reachy mic -> local Riva speech-to-text)

python3 ASR.py --seconds 5 --riva-uri 127.0.0.1:50051

Notes:
- Requires NVIDIA Riva server running (Jetson Orin Nano friendly).
- You must install the Riva Python client in the SAME python env you run ASR.py with.
	- Example (PyPI): pip install nvidia-riva-client
	- Or install the official wheel from the Riva quickstart/release bundle (version must match your server).
- Make sure the Riva gRPC port is reachable (default: 50051).

If you want to use the workspace virtualenv:

./.venv/bin/python src/ASR.py --seconds 5 --riva-uri 127.0.0.1:50051
