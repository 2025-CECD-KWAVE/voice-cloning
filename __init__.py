# -*- coding: utf-8 -*-
from flask import Flask, Response, request
import logging
from logging.handlers import RotatingFileHandler
from voice_cloning import get_tts_wav  
from io import BytesIO
import soundfile as sf
import time

app = Flask(__name__)

ref_wav_path = "ref_audio/ref_iu1.wav"
prompt_text = "사실 저는 아주 오랫동안 이 빨간색 가나 밀크를 제일 좋아했어요."

file_handler = RotatingFileHandler(
    'logs/server.log', maxBytes=2000, backupCount=10)
file_handler.setLevel(logging.ERROR)
app.logger.addHandler(file_handler)

formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)

@app.route('/iu', methods=['GET'])
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
    app.run()
    