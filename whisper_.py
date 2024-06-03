import speech_recognition
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from speech_recognition import Recognizer

recognizer_pers = Recognizer()
recognizer_pers.pause_threshold = 10 # время паузы в произношении после которого ПО распознает окончание предложения.

################################################################################################# ДАННАЯ ФУНКЦИЯ ВЫЗЫВАЕТЬСЯ ПРИ НАЖАТИИ НА КНОПКУ АУДИОЗАПИСИ. ################################################
def record_audio(audio): # Для данной нейроной модели предпочтительно использовать аудиосигнал в формате wav (я использую частоту дискретизауции 24000Гц по моим наблюдением это оптимальное соотношение между размером файла и качеством аудиозаписи, ОБЯЗАТЕЛЬНО mono -- по той причине, что стерео звук она обработать не сможет)
    with open('Audio.wav', 'wb') as file:
        wav_data = audio.get_wav_data()
        file.write(wav_data)
    data, samplerate = sf.read('Audio.wav')
    result = pipe(data, generate_kwargs={"language": "russian"})  # Если audio_waw не работае, то замениет его на data из закоментированного выше кода
    print(result["text"])
    return result["text"]



############################################################################### ВСТАВЬТЕ ЭТОТ КОД В НАЧАЛО РАБОТЫ ПРИЛОЖЕНИЯ, ЧТОБЫ МОДЕЛЬ ПОДГРУЖАЛАСЬ ВМЕСТЕ С ЗАПУСКОМ ПРИЛОЖЕНИЯ, А НЕ ПРИ КАЖДОМ ОБРАЩЕНИИ К НЕЙ ########################################################################################

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 #if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

models = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)  #Загружает модель с сайта HuggingFace вертуальное окружение
models.to(device)

processor = AutoProcessor.from_pretrained(model_id)


# Данная функция направляет аудио-файл на распознавание в нейроную сеть
pipe = pipeline(
    "automatic-speech-recognition",
    model=models,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
########################################################################################################################################################################################################################################################################################################################
