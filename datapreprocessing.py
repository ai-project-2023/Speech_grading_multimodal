
### Speech to text api
# ## 모듈 설치 코드
# !python -m pip install speechrecognition 

## import module
import speech_recognition as sr
from pydub import AudioSegment
import os

class Datapreprocessing():

    def __init__(self, data_path, data_name):
        self.data_path = data_path
        mp4_file_path = self.data_path + "mp4_file" + data_name ## format은 mp4로
        self.data = AudioSegment.from_file(mp4_file_path, format='mp4')
        # self.audio = None ## wav 음성 파일
    
    def mp4_to_wav(self, wav_name):
        # MP4 파일 로드
        wav_file_path = self.data_path + "wav_file" + wav_name
        # WAV 파일로 변환 -> path에 저장
        self.data.export(wav_file_path, format='wav')

    def batch_mp4_to_wav(self):
            # data_path 내의 모든 MP4 파일을 WAV 파일로 변환
            mp4_files = [f for f in os.listdir(self.data_path + "mp4_file") if f.endswith('.mp4')]
            for mp4_file in mp4_files:
                mp4_file_path = os.path.join(self.data_path, mp4_file)
                wav_name = os.path.splitext(mp4_file)[0] + ".wav"
                wav_file_path = os.path.join(self.data_path, wav_name)

                # MP4 파일 로드
                audio = AudioSegment.from_file(mp4_file_path, format='mp4')

                # WAV 파일로 변환
                audio.export(wav_file_path, format='wav')

    def speech_to_text(self):
        r = sr.Recognizer()
        wav_files = [f for f in os.listdir(os.path.join(self.data_path, "wav_file")) if f.endswith('.wav')]
        for wav_file in wav_files:
            wav_file_path = os.path.join(self.data_path, "wav_file", wav_file)
            with sr.AudioFile(wav_file_path) as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            
            # 결과를 파일에 저장
            result_file_path = os.path.join(self.data_path, "text_results", os.path.splitext(wav_file)[0] + ".txt")
            with open(result_file_path, "w") as file:
                file.write(text)
