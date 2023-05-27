import numpy as np
import matplotlib.pyplot as plt
import opensmile
import copy

class opensmile_wav():
      # 초기화
      def __init__(self, data_path: str):
        self.data_path = data_path
        self.data_features = None
        self.data_jitterLocal_sma3nz = None
        self.data_ShimmerLocaldB_sma3nz = None

      def wav_to_MFCC(self):
        ## good dataset
        wav_file = self.data_path
        # OpenSMILE 추출기 초기화
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        # WAV 파일에서 MFCC 추출
        features = smile.process_file(wav_file)
        self.data_features = copy.copy(features)

      def jitterLocal_sma3nz(self):
        # jitterLocal_sma3nz
        jitter_values = self.data_features['jitterLocal_sma3nz'].values
        self.data_jitterLocal_sma3nz = jitter_values 

      def ShimmerLocaldB_sma3nz(self):
      # ShimmerLocaldB_sma3nz 
        shimmer_values = self.data_features['shimmerLocaldB_sma3nz'].values
        # self.data_ShimmerLocaldB_sma3nz = copy.copy(shimmer_values) 
        self.data_ShimmerLocaldB_sma3nz = shimmer_values 


      def show_result(self, res_type:str):
        if res_type == 'ShimmerLocaldB':
          if self.data_ShimmerLocaldB_sma3nz is None:
            print('run ShimmerLocaldB_sma3nz(self)')
          else:
            plt.specgram(self.data_ShimmerLocaldB_sma3nz, Fs=1000, cmap='jet')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.title('Spectrogram of' + res_type)
            plt.colorbar(format='%+2.0f dB')
            plt.show()
        elif res_type == 'jitterLocal':
          if self.data_jitterLocal_sma3nz is None:
            print('run data_jitterLocal_sma3nz(self)')
          else:
            plt.specgram(self.data_jitterLocal_sma3nz, Fs=1000, cmap='jet')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.title('Spectrogram of' + res_type)
            plt.colorbar(format='%+2.0f dB')
            plt.show()
        else:
          print("not exist type")


