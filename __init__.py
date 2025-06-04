# -*- coding: utf-8 -*-
from flask import Flask, Response, request
import logging
from logging.handlers import RotatingFileHandler
from io import BytesIO
import soundfile as sf
import os, gdown

app = Flask(__name__)

sovits_path = "SoVITS_weights_v2/IUv2_e8_s216.pth"
gpt_path = "GPT_weights_v2/IUv2-e15.ckpt"
sovits_download_url = "https://drive.google.com/uc?id=1SaIil6qaD7T1XhLinyXAnUvIWliXc5dk"
gpt_download_url = "https://drive.google.com/uc?id=1ERpGWMwUZFrswyIyHDER36cWxkxIjGwk"

roberta_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin"
roberta_download_url = "https://drive.google.com/uc?id=1LW4I17dun5ZJwpkYYERVmlqRxeg0foZR"
hubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin"
hubert_download_url = "https://drive.google.com/uc?id=1pLzRY4DMXeIm36I-FrPnKRx3JPn7ZKp5"

ref_wav_path = "ref_audio/ref_iu1.wav"
prompt_text = "사실 저는 아주 오랫동안 이 빨간색 가나 밀크를 제일 좋아했어요."

def file_check(path, download_url):
    if not os.path.exists(path):
        if not os.path.exists(path.replace(path.split("/")[-1], "")):
            os.makedirs(path.replace(path.split("/")[-1], ""))
        gdown.download(download_url, path, quiet=False)

file_check(sovits_path, sovits_download_url)
file_check(gpt_path, gpt_download_url)
file_check(roberta_path, roberta_download_url)
file_check(hubert_path, hubert_download_url)

if not os.path.exists("logs/"):
    os.mkdir("logs/")

file_handler = RotatingFileHandler(
    'logs/server.log', maxBytes=2000, backupCount=10)
file_handler.setLevel(logging.ERROR)
app.logger.addHandler(file_handler)

formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)

from voice_cloning import get_tts_wav  
@app.route('/iu', methods=['POST'])
def example():
    try:
        data = request.get_json()
        language = data['language']
        text = data['text']
        
        app.logger.debug(f'tts text: {text}')
        sr, audio_opt = get_tts_wav(ref_wav_path, prompt_text, "한국어", text, language, top_k=15, top_p=1, temperature=1, sample_steps=32)
        buffer = BytesIO()
        sf.write(buffer, audio_opt, sr, format='WAV')
        buffer.seek(0)
        audio = buffer.read()
        
    except Exception as e:
        print(e)
        app.logger.error(e)
    
    return Response(audio, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5333)
    