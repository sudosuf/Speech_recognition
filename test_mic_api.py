import pyaudio
import requests
import speech_recognition as sr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
import numpy as np
import logging
from typing import List

# Create structure log file
logging.basicConfig(level=logging.INFO, filename="py_log.log", filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger_record = logging.getLogger("Recognition_log")

# FastAPI app initialization
try:
    app = FastAPI()
except Exception as e:
    logger.error(f"Failed to start FastAPI: {e}")
    raise

# Global variables to store the results
Audio_data = []
PARTICAL_TEXT = ""
NAME_AUDIO = 'Audio.wav'
BOOL_LISTEN = True
RECOGNIZE_TEXT = ""

sr.Recognizer.pause_threshold = 7

# Initialize recognizer
recognizer = sr.Recognizer()

# Initialize PyAudio
p = pyaudio.PyAudio()

# Audio stream configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

STOP_LISTENING = True

# Open PyAudio stream
try:
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    logger.info("PyAudio stream open successfully")
except Exception as e:
    logger.error(f"Failed to open PyAudio stream: {e}")
    raise

# Initialize Whisper model
try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v2"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    logger.info("Whisper model and pipeline initialize successfully")
except Exception as e:
    logger.error(f"Failed to initialize Whisper model and pipeline: {e}")
    raise


# Function to recognize partial speech
def partical_recognize():
    global BOOL_LISTEN, PARTICAL_TEXT, Audio_data
    try:
        while BOOL_LISTEN:
            data = stream.read(CHUNK)
            Audio_data.append(data)
            audio_chunk = np.frombuffer(data, np.int16)
            if len(data) > 0:
                audio_chunk = np.float32(audio_chunk) / 32768.0  # Normalize to range [-1, 1]
                results = pipe(audio_chunk, generate_kwargs={"language": "russian"})
                text = results["text"]
                PARTICAL_TEXT += text
                print('PARTIAL TEXT:', PARTICAL_TEXT)
            else:
                logger.warning(f"Len(data) <= 0: {len(data)}")
    except Exception as e:
        logger.error(f"Error during partial recognition : {e}")
        raise


def full_recognize():
    global RECOGNIZE_TEXT, Audio_data
    try:
        full_audio = b''.join(Audio_data)
        full_audio = np.frombuffer(full_audio, np.int16)
        full_audio = np.float32(full_audio) / 32768.0  # Normalize to range [-1, 1]
        results = pipe(full_audio, generate_kwargs={"language": "russian"})
        RECOGNIZE_TEXT = results["text"]
        print('FINAL TEXT:', RECOGNIZE_TEXT)
    except Exception as e:
        logger.error(f"Error during full recognition: {e}")
        raise


def callback(recognizer, audio_data):
    global BOOL_LISTEN
    try:
        BOOL_LISTEN = False
        full_recognize()
    except Exception as e:
        logger.error(f"Error in callback: {e}")
        raise


# Start PyAudio stream
try:
    stream.start_stream()
    logger.info("Pyaudio start_stream successfully")
except Exception as e:
    logger.error(f"Failed to start_stream PyAudio: {e}")
    raise

# Create audio source for recognizer
try:
    audio_source = sr.Microphone()
    logger.info("Audio source for recognition created successfully")
except Exception as e:
    logger.error(f"Failed to create audio source: {e}")
    raise


# Listen in background


class AudioInput(BaseModel):
    audio_data: bytes


@app.post("/recognize/")
async def recognize_speech(audio: AudioInput, background_tasks: BackgroundTasks, request: Request):
    global BOOL_LISTEN, Audio_data, PARTICAL_TEXT, STOP_LISTENING
    try:
        ip_client = request.client.host
        logger.info(f"Recognition task started from IP: {ip_client}")
        logger_record.info(f"Recognition task started from IP: {ip_client}")
        BOOL_LISTEN = True
        Audio_data = []
        PARTICAL_TEXT = ""
        background_tasks.add_task(partical_recognize)
        return {"status": "Listening for speech"}
    except Exception as e:
        ip_client = request.client.host
        logger.error(f"Error in /recognize endpoint from IP: {ip_client}: {e}")
        return {"status": "Error", "message": str(e)}


@app.get("/result/")
async def get_recognition_result(request: Request):
    global PARTICAL_TEXT, RECOGNIZE_TEXT, BOOL_LISTEN
    try:
        if BOOL_LISTEN:
            return {"partial_text": PARTICAL_TEXT}
        else:
            ip_client = request.client.host
            logger.info(f"Returning final recognition result to IP: {ip_client}")
            logger_record.info(f"F_R: {RECOGNIZE_TEXT}")
            return {"final_text": RECOGNIZE_TEXT}
    except Exception as e:
        ip_client = request.client.host
        logger.error(f"Error in /result endpoint from IP: {ip_client}: {e}")
        return {"status": "Error", "message": str(e)}


# Stop the background listening and close the audio stream
@app.on_event("shutdown")
def shutdown_event():
    try:
        stream.stop_stream()
        stream.close()
        p.terminate()
        logger.info("Shutdown event: closed all stream and terminated PyAudio")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Run FastAPI
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="192.168.10.1", port=6464)
        logger.info("start FastAPI application successfully")
    except Exception as e:
        logger.error(f"Failed to start FastAPI application: {e}")
        raise
