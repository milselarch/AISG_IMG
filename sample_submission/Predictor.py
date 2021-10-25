import time
import torch
import misc

import pandas as pd
import subprocess
import numpy as np
import nvidia_smi
import os

from argparse import ArgumentParser
from numpy import random
from tqdm.auto import tqdm
from PIL import Image

from AISG.loader import load_video
from AISG.DeepfakeDetection.FaceExtractor import FaceExtractor
from AISG.MesoNet.MesoTrainer import MesoTrainer
from AISG.FakeVoiceTorch.trainer import Trainer as AudioPredictor
from AISG.NeuralFaceExtract import NeuralFaceExtract

from Timer import Timer
from PredictionsHolder import PredictionsHolder

vram_gb = misc.get_gpu_capacity()
BIG_GPU = True if vram_gb > 12 else False
print(f'GPU VRAM = {vram_gb}GB, USE-GPU [{BIG_GPU}]')
print('VERSION 0.0.8')

class Predictor(object):
    def __init__(self):
        self.face_all_timer = Timer()
        self.face_predict_timer = Timer()

        self.audio_predictor = AudioPredictor(
            preload_path='models/AUD-211002-1735.pt',
            cache_threshold=20, train_version=1,
            use_batch_norm=True, add_aisg=False, use_avs=True,
            load_dataset=False, use_cuda=BIG_GPU
        )
        self.face_predictor = MesoTrainer(
            preload_path='models/MES-211022-0001.pt',
            load_dataset=False, use_cuda=BIG_GPU
        )

        self.preds_holder = None
        self.face_extractor = None
        # print(os.system('ls -a'))
        # print(os.system('ls /data/input -a'))
        # input('TEST ')

    @staticmethod
    def extract_audios(test_videos, input_dir, out_dir):
        if input_dir.endswith('/'):
            input_dir = input_dir[:-1]

        pbar = tqdm(test_videos)

        for filename in pbar:
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
        transform = self.face_predictor.transform
        # print('MAP FILEPATH =', filepath)
        name = misc.path_to_name(filepath)
        filename = f'{name}.mp4'
        per_face_pred = []

        for face_no in face_image_map:
            face_images = face_image_map[face_no]
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

            face_pred = np.percentile(sorted(preds), 75)
            print(f'F-PRED {face_pred}')
            per_face_pred.append(face_pred)

        if len(per_face_pred) != 0:
            face_pred = max(per_face_pred)
        else:
            print(f'FACELESS {filename}')
            face_pred = 0.85

        print(f'ADD POP RESULT', filepath)
        # pbar.n = extractor.completed
        status = f'FACE PRED [{filename}] = {face_pred}'
        pbar.set_description(status)
        self.preds_holder.add_face_pred(filename, face_pred)

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

        test_videos = [
            video for video in os.listdir(input_dir)
            if ".mp4" in video
        ]

        print(f'SHOWING ALL VIDEOS')
        for k, filename in enumerate(test_videos):
            print(f'[{k}] - [{filename}]')

        self.extract_audios(test_videos, input_dir, temp_dir)

        self.face_all_timer.start()
        self.face_extractor = NeuralFaceExtract()
        self.face_extractor.process_filepaths(
            test_videos, every_n_frames=20, batch_size=16,
            callback=self.handle_video_face_preds,
            base_dir=input_dir
        )

        self.face_all_timer.pause()

        audio_pbar = tqdm(test_videos)
        for filename in audio_pbar:
            name = misc.path_to_name(filename)
            audio_filepath = f'{temp_dir}/{name}.flac'
            audio_preds = self.audio_predictor.batch_predict(
                audio_filepath
            )

            audio_preds = audio_preds.flatten()
            audio_pred = np.median(audio_preds)
            self.preds_holder.add_audio_pred(filename, audio_pred)
            status = f'AUD PRED [{name}] = {audio_pred}'
            audio_pbar.set_description(status)

        print(f'face time: {self.face_all_timer.total}')
        print(f'face predict time: {self.face_predict_timer.total}')
        self.preds_holder.export()