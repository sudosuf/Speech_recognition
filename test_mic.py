import pyaudio
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import soundfile as sf
from scipy.io.wavfile import write
import numpy

import json
import time
import wave




#####################################################################################
"""Глобальные переменые. Результат работы кода будет заноситься в эти переменые"""
Audio_data=0
PARTICAL_TEXT = "" # частично распознаный текст
NAME_AUDIO = 'Audio.wav'
BOOL_LISTEN = True # сигнал о работе по частичному распознаванию текста, когда человек закончит говорить, система присвоит данной переменной  начение False .
                   # после этого можно считывать переменую RECOGNIZE_TEXT
RECOGNIZE_TEXT = "" # целиком распознаный текст
#####################################################################################





sr.Recognizer.pause_threshold = 7 # Настройка паузы после которой система перестанет слушать

# Инициализация распознавателя речи
recognizer = sr.Recognizer()

# Инициализация PyAudio
p = pyaudio.PyAudio()

# Конфигурация аудио потока
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Открываем поток PyAudio
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Инициализация KaldiRecognizer
model = Model("vosk-model-small-ru-0.22")  # Убедитесь, что путь к модели указан правильно
kaldi_recognizer = KaldiRecognizer(model, RATE)

def Partical_recognize():
    while True:
        if BOOL_LISTEN == True:
            data = stream.read(16000)
            if len(data) == 0:
                break
            else:
                # write('test.wav', RATE, numpy.float32(data))
                # data, _ = sf.read('Audio_tmp.wav')
                # text = pipe(data)
                # print("textttttttttttt  ", data)
                text = kaldi_recognizer.PartialResult()[14:-3]
                PARTICAL_TEXT.join(text)
                print('TEXT: ', text)
        else:
            break
        # print(rec.Result() if rec.AcceptWaveform(data) else rec.PartialResult())
    # print('final: ', kaldi_recognizer.FinalResult())


def callback(recognizer, audio): # Сюда передаются аудио данные записанные с SpeechRecognition.Recognizer.listen_in_background()
    try:
        # Записывае одно аудио ведь мы распознаем только одну часть речи, без привязки к другим запросам
        with open(NAME_AUDIO, 'wb') as file:
            #print(audio.get_wav_data())
            file.write(audio.get_wav_data())

        data, samp = sf.read(NAME_AUDIO)

        global Audio_data
        Audio_data = data

        rezult = pipe(data, generate_kwargs={"language": "russian"})

        print("SpeechRecognition: ", rezult)
        global RECOGNIZE_TEXT
        RECOGNIZE_TEXT = rezult

        global BOOL_LISTEN
        BOOL_LISTEN = False

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


# Открываем поток для Kaldi
stream.start_stream()
# Создаем аудио источник для распознавателя
audio_source = sr.Microphone()





################################### ЗАПУСКАЕМ WHISPER ###########################################
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

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
################################################################################################




# Начинаем прослушивание микрофон в фоновом режине (в новом потоке)
stop_listening = recognizer.listen_in_background(audio_source, callback)


# Основной цикл программы
try:
    while True:
        if BOOL_LISTEN:
            Partical_recognize()
            BOOL_LISTEN = True
        else:
            break

except KeyboardInterrupt:
    print("Программа завершена.")
finally:
    # Останавливаем фоновое прослушивание
    stop_listening(wait_for_stop=False)
    stream.stop_stream()
    stream.close()
    p.terminate()

