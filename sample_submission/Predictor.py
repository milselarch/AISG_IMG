import time
import torch
import misc

import torch.multiprocessing as mp
import pandas as pd
import subprocess
import numpy as np
import nvidia_smi
import pickle
import os

from sklearn.linear_model import LogisticRegression
from torch.utils import data as data_utils
from queue import Empty as EmptyQueue
from argparse import ArgumentParser
from numpy import random
from tqdm.auto import tqdm
from PIL import Image

from AISG.wav2lip import audio
from AISG.wav2lip.SyncnetTrainer import SyncnetTrainer
from AISG.loader import load_video
from AISG.DeepfakeDetection.FaceExtractor import FaceExtractor
from AISG.MesoNet.MesoTrainer import MesoTrainer
from AISG.FakeVoiceTorch.trainer import Trainer as AudioPredictor
from AISG.NeuralFaceExtract import NeuralFaceExtract

from Timer import Timer
from PredictionsHolder import PredictionsHolder
from Dataset import VideoDataset

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

vram_gb = misc.get_gpu_capacity()
BIG_GPU = True if vram_gb > 12 else False
FACE_BATCH_SIZE = 64 if BIG_GPU else 16
NUM_WORKERS = 2 if BIG_GPU else 2
print(f'GPU VRAM = {vram_gb}GB, USE-GPU [{BIG_GPU}]')
print('VERSION 0.1.0')

class Predictor(object):
    def __init__(self):
        self.timer = Timer()
        self.face_predict_timer = Timer()
        self.sync_predict_timer = Timer()
        self.audio_predict_timer = Timer()
        self.audio_file_queue = mp.Queue()

        self.audio_predictor = AudioPredictor(
            preload_path='models/AUD-211002-1735.pt',
            cache_threshold=20, train_version=1,
            use_batch_norm=True, add_aisg=False, use_avs=True,
            load_dataset=False, use_cuda=BIG_GPU
        )
        """
        self.face_predictor = MesoTrainer(
            preload_path='models/MES-211022-0001.pt',
            load_dataset=False, use_cuda=BIG_GPU
        )
        """
        self.face_predictor = MesoTrainer(
            preload_path='models/E16296960_T0.87_V0.88.pt',
            load_dataset=False, use_cuda=BIG_GPU,
            use_inception=True
        )
        self.sync_predictor = SyncnetTrainer(
            use_cuda=BIG_GPU, load_dataset=False, use_joon=True,
            preload_path='models/syncnet_joon.model',
            is_checkpoint=False, strict=False
        )
        self.sync_regressor = pickle.load(open(
            'models/logistic-sync.sav', 'rb'
        ))

        self.preds_holder = None
        self.face_extractor = None
        # print(os.system('ls -a'))
        # print(os.system('ls /data/input -a'))
        # input('TEST ')

    @staticmethod
    def extract_audios(
        test_videos, input_dir, out_dir,
        audio_file_queue
    ):
        if input_dir.endswith('/'):
            input_dir = input_dir[:-1]

        pbar = tqdm(test_videos)

        for k, filename in enumerate(pbar):
            pbar.set_description(f'extracting {filename}')
            filepath = f'{input_dir}/{filename}'
            name = filename[:filename.index('.')]

            out_path = f'{out_dir}/{name}.flac'
            ffmpeg_args = '-ab 160k -y -ac 2 -ar 44100'
            cmd = f'ffmpeg -i {filepath} {ffmpeg_args} -vn {out_path}'
            result = subprocess.run(
                cmd, capture_output=True, shell=True
            )

            if result.returncode != 0:
                print(f'ERROR CONVERTING {filename}')
                print(result.stderr.decode())

            audio_file_queue.put(filename)
            print(f'LOAD {k} {filename}')

        audio_file_queue.put('END')
        print('EXTRACTION DONE')

    def handle_face_preds(self, pbar):
        while True:
            result = extractor.pop()

            if result is None:
                # print('RESULT IS NONE')
                return

            filepath, face_image_map = result
            self.handle_video_face_preds(
                filepath, face_image_map, pbar
            )

    def handle_video_face_preds(
        self, filepath, face_image_map, pbar
    ):
        # print('MAP FILEPATH =', filepath)
        name = misc.path_to_name(filepath)
        filename = f'{name}.mp4'

        face_pred = self.predict_faces(face_image_map)

        print(f'ADD POP RESULT', filepath)
        # pbar.n = extractor.completed
        status = f'FACE PRED [{filename}] = {face_pred}'
        pbar.set_description(status)
        self.preds_holder.add_face_pred(filename, face_pred)

    @staticmethod
    def show_filenames(test_videos):
        print(f'SHOWING ALL VIDEOS')
        for k, filename in enumerate(test_videos):
            print(f'[{k}] - [{filename}]')

    @staticmethod
    def get_test_videos(input_dir):
        test_videos = [
            video for video in os.listdir(input_dir)
            if ".mp4" in video
        ]
        return test_videos

    def fetch_filename(self):
        while True:
            try:
                filename = self.audio_file_queue.get_nowait()
                if filename is not None:
                    return filename
            except EmptyQueue:
                pass

            time.sleep(1)

    def extract_face_preds(self, test_videos, input_dir):
        pbar = tqdm(range(len(test_videos)))

        for k in pbar:
            filename = self.fetch_filename()
            pbar.update()

            print(f'FF - [{k}/{len(test_videos)}] - {filename}')
            face_image_map = self.face_extractor.process_filepath(
                filepath=filename, base_dir=input_dir,
                every_n_frames=10, batch_size=FACE_BATCH_SIZE,
            )

            self.handle_video_face_preds(
                filepath=filename, face_image_map=face_image_map,
                pbar=pbar
            )

    def predict_audio(self, audio_arr):
        # print(type(audio_arr))
        if isinstance(audio_arr, torch.Tensor):
            audio_arr = audio_arr.numpy()

        self.audio_predict_timer.start()
        audio_preds = self.audio_predictor.predict_raw(audio_arr)
        self.audio_predict_timer.pause()
        audio_preds = audio_preds.flatten()
        audio_pred = np.median(audio_preds)
        return audio_pred

    def predict_faces(self, face_image_map):
        transform = self.face_predictor.transform
        per_face_pred = []

        for face_no in face_image_map:
            face_images = face_image_map.get_detected_frames(face_no)
            torch_images = []

            for frame_no in face_images:
                face = face_images[frame_no]
                face_image = face.image

                pil_image = Image.fromarray(face_image)
                torch_image = transform(pil_image)
                torch_image = torch.unsqueeze(torch_image, 0)
                torch_images.append(torch_image)

            torch_batch = torch.cat(torch_images, 0)
            self.face_predict_timer.start()
            preds = self.face_predictor.batch_predict(torch_batch)
            self.face_predict_timer.pause()

            face_pred = np.percentile(sorted(preds), 25)
            print(f'Q1 FACE PRED {face_pred}')
            per_face_pred.append(face_pred)

        if len(per_face_pred) != 0:
            face_pred = max(per_face_pred)
        else:
            print(f'FACELESS {filename}')
            face_pred = 0.85

        return face_pred

    def predict_sync(self, face_image_map, audio_array):
        if len(face_image_map) == 0:
            return 0.5

        per_face_pred = []

        for face_no in face_image_map:
            face_samples = face_image_map.sample_face_frames(
                face_no, consecutive_frames=5, extract=False
            )
            # print(f'FACE SAMPLES, {face_samples}')
            self.sync_predict_timer.start()
            distances = self.sync_predictor.face_predict_joon(
                face_samples, audio_array, fps=face_image_map.fps,
                to_numpy=True, is_raw_audio=True
            )

            self.sync_predict_timer.pause()

            mean_pred = np.mean(distances)
            median_pred = np.median(distances)
            quartile_pred_3 = np.percentile(sorted(distances), 75)
            quartile_pred_1 = np.percentile(sorted(distances), 25)
            pred_batch = np.array([
                mean_pred, median_pred, quartile_pred_1,
                quartile_pred_3
            ])

            per_face_pred.append(pred_batch)

        if len(per_face_pred) == 0:
            return 0.

        sync_preds = np.min(per_face_pred, axis=0)
        sync_pred = self.sync_regressor.predict_proba([sync_preds])
        sync_pred = sync_pred[0][1]
        sync_pred = sync_pred * 0.95 - 0.05
        return sync_pred

    def main(self, input_dir, output_file, temp_dir=None):
        output_dir = output_file[:output_file.rindex('/')]
        input_dir = input_dir[:output_file.rindex('/')]
        print(f'output dir {output_dir}')

        if temp_dir is None:
            temp_dir = f'{output_dir}/temp'
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        self.preds_holder = PredictionsHolder(input_dir, output_file)
        # read input directory for mp4 videos only
        # note: all files would be mp4 videos in the mounted input dir

        print(f'INPUT DIR {input_dir}')
        test_videos = self.get_test_videos(input_dir)
        self.show_filenames(test_videos)

        audio_extract_process = mp.Process(
            target=self.extract_audios,
            kwargs=misc.kwargify(
                test_videos=test_videos, input_dir=input_dir,
                audio_file_queue=self.audio_file_queue,
                out_dir=temp_dir
            )
        )

        audio_extract_process.start()
        # audio_extract_process.join()
        # self.extract_audios(test_videos, input_dir, temp_dir)
        # self.face_extractor = NeuralFaceExtract()
        # self.extract_face_preds(test_videos, input_dir)
        print('EXTRACTION STARTED')

        video_dataset = VideoDataset(
            file_queue=self.audio_file_queue,
            num_files=len(test_videos), input_dir=input_dir,
            temp_dir=temp_dir, face_batch_size=FACE_BATCH_SIZE,
            # face_extractor=self.face_extractor
        )

        print(f'NUM WORKERS {NUM_WORKERS}')
        data_loader = data_utils.DataLoader(
            video_dataset, batch_size=None, num_workers=NUM_WORKERS,
            pin_memory=False
        )

        k = 0
        num_videos = len(test_videos)
        pbar = tqdm(range(num_videos))
        self.timer.start()

        for sample in data_loader:
            if sample is None:
                print('END SAMPLE')
                continue

            filename, audio_array, face_image_map = sample
            audio_pred = self.predict_audio(audio_array)
            face_pred = self.predict_faces(face_image_map)
            sync_pred = self.predict_sync(face_image_map, audio_array)
            # sync_pred = 0

            video_pred = max(
                audio_pred, face_pred, sync_pred
            )

            self.preds_holder.add_pred(filename, video_pred)
            print(f'PREDICTING [{k}] [{filename}]')
            print(f'AUD-PRED = {audio_pred}')
            print(f'FACE-PRED = {face_pred}')
            print(f'SYNC-PRED = {sync_pred}')

            face_status = f'FP={face_pred:2f}'
            audio_status = f'AP={audio_pred:2f}'
            sync_status = f'SP={sync_pred:2f}'

            stats = f'{face_status}, {audio_status}, {sync_status}'
            desc = f'[{k+1}/{num_videos}] [{filename}] {stats}'
            pbar.set_description(desc)
            pbar.update()
            k += 1

        self.timer.pause()
        print(f'total predict time: {self.timer.total}')
        print(f'face predict time: {self.face_predict_timer.total}')
        print(f'sync predict time: {self.face_predict_timer.total}')
        print(f'audio predict time: {self.audio_predict_timer.total}')
        self.preds_holder.export()


if __name__ == "__main__":
    mp.freeze_support()