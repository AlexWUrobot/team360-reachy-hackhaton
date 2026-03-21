import time

import cv2
from google import genai
from google.genai import types
from reachy_mini import ReachyMini

PROMPT = """
         Point to no more than 10 items in the image. The label returned
         should be an identifying name for the object detected.
         The answer should follow the json format: [{"point": <point>,
         "label": <label1>}, ...]. The points are in [y, x] format
         normalized to 0-1000.
        """
client = genai.Client()

with ReachyMini() as mini:
    print("Connecting to Reachy and waking up camera...")
    time.sleep(1.5)

    frame = mini.media.get_frame()

    if frame is None:
        print(
            "Error: Reachy failed to grab a frame. Please check if the camera is connected or if macOS camera permissions are blocking the terminal."
        )
        exit(1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_bytes = cv2.imencode(".png", rgb_frame)[1].tobytes()

print("Frame captured! Sending to Gemini...")

image_response = client.models.generate_content(
    model="gemini-robotics-er-1.5-preview",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",
        ),
        PROMPT,
    ],
    config=types.GenerateContentConfig(
        temperature=0.5, thinking_config=types.ThinkingConfig(thinking_budget=0)
    ),
)

print(image_response.text)
