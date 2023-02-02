from pydub import AudioSegment
import math
from glob import glob

train_path_wav=glob("C:/Users/hshla/Desktop/audio/*")
for train_wav_all in train_path_wav:
    train_wav=glob(train_wav_all+"/*")
    for wav_file in train_wav:
        wav_file=wav_file.split("\\")
        print(wav_file[-1])
        # song = AudioSegment.from_wav(wav_file)
        #
        #
        # sixty_seconds = 1000
        #
        # for i in range(int(math.floor(len(song)/60000))):
        #     slice = song[i*sixty_seconds:sixty_seconds*(i+1)]
        #     slice.export('newSong_{}.mp3'.format(i), format="wav")