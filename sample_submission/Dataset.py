import time

import cv2
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
        temp_dir, face_batch_size, face_extractor=None,
        use_cuda=True
    ):
        self.file_queue = file_queue
        self.face_batch_size = face_batch_size
        self.input_dir = input_dir
        self.temp_dir = temp_dir
        self.num_files = num_files

        if face_extractor is None:
            self.face_extractor = NeuralFaceExtract(
                use_cuda=use_cuda
            )
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
        filepath = f'{self.input_dir}/{filename}'
        cap = cv2.VideoCapture(filepath)

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale = 0.5
        if min(width_in, height_in) < 700:
            scale = 1

        if num_frames <= 10 * 24:
            skip_detect = 5
        elif num_frames >= 30 * 24:
            skip_detect = 20
        else:
            skip_detect = 10

        face_image_map = self.face_extractor.process_video(
            cap, batch_size=self.face_batch_size,
            every_n_frames=1, skip_detect=skip_detect,
            ignore_detect=5, scale=scale, filepath=filepath
        )

        audio_filepath = f'{self.temp_dir}/{name}.flac'
        audio_array, sample_rate = librosa.load(
            audio_filepath, sr=16000
        )

        print('LOADED', filename, rand, qsize)
        return filename, audio_array, face_image_map


if __name__ == '__main__':
    mp.freeze_support()
