import os
from pydub import AudioSegment
import speech_recognition as sr
# API = "YOUR_API_KEY"

class Datapreprocessing:
    def __init__(self, data_path):
        self.data_path = data_path
        self.mp4_file_path = os.path.join(self.data_path, "mp4_file")
        self.wav_file_path = os.path.join(self.data_path, "wav_file")
        self.text_results_path = os.path.join(self.data_path, "text_results")

    def mp4_to_wav(self, wav_name, mp4_file):
        mp4_file_path = os.path.join(self.mp4_file_path, mp4_file)
        wav_file_path = os.path.join(self.wav_file_path, wav_name)
        audio = AudioSegment.from_file(mp4_file_path, format='mp4')
        audio.export(wav_file_path, format='wav')

    def batch_mp4_to_wav(self):
        mp4_files = [f for f in os.listdir(self.mp4_file_path) if f.endswith('.mp4')]
        for mp4_file in mp4_files:
            wav_name = os.path.splitext(mp4_file)[0] + ".wav"
            self.mp4_to_wav(wav_name, mp4_file)

    def speech_to_text(self):
        r = sr.Recognizer()
        wav_files = [f for f in os.listdir(self.wav_file_path) if f.endswith('.wav')]
        for wav_file in wav_files:
            wav_file_path = os.path.join(self.wav_file_path, wav_file)
            with sr.AudioFile(wav_file_path) as source:
                audio = r.record(source)
            try: 
              text = r.recognize_google(audio)
              result_file_path = os.path.join(self.text_results_path, os.path.splitext(wav_file)[0] + ".txt")
              with open(result_file_path, "w") as file:
                  file.write(text)
            except:
              print("running error")



data_path = os.path.abspath("./drive/MyDrive/datapreprocessing/dataset")
preprocessing = Datapreprocessing(data_path)
preprocessing.batch_mp4_to_wav()
preprocessing.speech_to_text()
