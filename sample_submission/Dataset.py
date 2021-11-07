import time
import librosa
import torch
import random
import torch.multiprocessing as mp

from torch.multiprocessing import set_start_method
from queue import Empty as EmptyQueue

from AISG.NeuralFaceExtract import NeuralFaceExtract

try:
    set_start_method('spawn')
except RuntimeError:
    pass

class VideoDataset(object):
    def __init__(
        self, file_queue, num_files, input_dir,
        temp_dir, face_batch_size, face_extractor=None
    ):
        self.file_queue = file_queue
        self.face_batch_size = face_batch_size
        self.input_dir = input_dir
        self.temp_dir = temp_dir
        self.num_files = num_files

        if face_extractor is None:
            self.face_extractor = NeuralFaceExtract()
        else:
            self.face_extractor = face_extractor

    def __len__(self):
        return self.num_files

    def fetch_filename(self):
        turns = 0

        while True:
            try:
                filename = self.file_queue.get_nowait()
                if filename is not None:
                    return filename
            except EmptyQueue:
                pass

            # print('WAITING', self.file_queue.qsize())
            time.sleep(1)
            turns += 1

    def __getitem__(self, idx):
        rand = random.random()
        qsize = self.file_queue.qsize()
        filename = self.fetch_filename()
        print('QSIZE', qsize, rand)

        if filename == 'END':
            print('FETCH END', rand)
            return None

        assert filename.endswith('.mp4')
        print('filename', filename)

        name = filename[:filename.index('.')]
        face_image_map = self.face_extractor.process_filepath(
            filepath=filename, base_dir=self.input_dir,
            batch_size=self.face_batch_size,
            every_n_frames=1, skip_detect=10, ignore_detect=5
        )

        audio_filepath = f'{self.temp_dir}/{name}.flac'
        audio_array, sample_rate = librosa.load(
            audio_filepath, sr=16000
        )

        print('LOADED', filename, rand, qsize)
        return filename, audio_array, face_image_map


if __name__ == '__main__':
    mp.freeze_support()
