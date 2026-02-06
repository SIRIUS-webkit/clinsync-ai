import requests
import io
from PIL import Image
import wave
import json


def create_dummy_image():
    # Create a 100x100 white image
    img = Image.new("RGB", (100, 100), color="white")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr


def create_dummy_audio():
    # Create 1 second of silence
    audio_byte_arr = io.BytesIO()
    with wave.open(audio_byte_arr, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00" * 32000)
    audio_byte_arr.seek(0)
    return audio_byte_arr


def test_chat_api():
    url = "http://localhost:8000/chat/"

    # Prepare files
    image_file = create_dummy_image()
    audio_file = create_dummy_audio()

    files = [
        ("image", ("test_image.jpg", image_file, "image/jpeg")),
        ("audio", ("test_audio.wav", audio_file, "audio/wav")),
    ]

    data = {
        "text": "This is a test request from the standalone backend test script.",
        "patient_id": "test_patient_123",
    }

    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, data=data, files=files)

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error Response:")
            print(response.text)

    except Exception as e:
        print(f"Failed to connect: {e}")


if __name__ == "__main__":
    test_chat_api()
