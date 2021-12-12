import functools
import sys
import time
import torch
import misc

import torch
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import subprocess
import nvidia_smi
import pickle
import sys
import os

from sklearn.linear_model import LogisticRegression
from torch.utils import data as data_utils
from queue import Empty as EmptyQueue
from tqdm.auto import tqdm as raw_tqdm
from argparse import ArgumentParser
from numpy import random
from PIL import Image

from AISG.wav2lip import audio
from AISG.wav2lip.FaceSamplesHolder import FaceSamplesHolder
from AISG.wav2lip.SyncnetTrainer import SyncnetTrainer
from AISG.FaceImageMap import FaceImageMap
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

tqdm = functools.partial(raw_tqdm, file=sys.stdout)

class Predictor(object):
    def __init__(self, seed=42):
        self.timer = Timer()
        self.face_predict_timer = Timer()
        self.sync_predict_timer = Timer()
        self.audio_predict_timer = Timer()
        self.audio_file_queue = mp.Queue()
        self.use_mouth_image = True

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
            preload_path='models/MES_E16296960_T0.87_V0.88.pt',
            load_dataset=False, use_cuda=BIG_GPU,
            use_inception=True
        )

        self.sync_rgb_swap = True

        if self.use_mouth_image:
            self.sync_predictor = SyncnetTrainer(
                use_cuda=BIG_GPU, load_dataset=False,
                use_joon=True, old_joon=False,
                preload_path='models/SYNM_E10695968_T0.84_V0.69.pt',

                fcc_list=(512, 128, 32),
                pred_ratio=1.0, dropout_p=0.5,
                is_checkpoint=False, predict_confidence=True,
                transform_image=True, eval_mode=False
            )
        else:
            self.sync_predictor = SyncnetTrainer(
                use_cuda=BIG_GPU, load_dataset=False,
                is_checkpoint=False, strict=False,
                use_joon=True, old_joon=False, pred_ratio=1.0,
                fcc_list=(512, 128, 32), dropout_p=0.5,
                eval_mode=True,

                preload_path='models/SYNF_E6143040_T0.77_V0.66.pt',
                transform_image=False, predict_confidence=False
            )

        self.preds_holder = None
        self.face_extractor = None
        self.face_batch_size = 32

        self.dataset_use_cuda = False
        self.seed = seed

        random.seed(seed)
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
            pbar.set_description('extracting', filename)
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
        audio_preds = self.audio_predictor.predict_raw(
            audio_arr, no_grad=True, max_batch_size=6,
            min_samples=256, clip_p=0
        )
        """
        audio_preds = self.audio_predictor.predict_raw(
            audio_arr, max_samples=512, min_samples=256,
            clip_p=0.05
        )
        """
        self.audio_predict_timer.pause()
        audio_preds = audio_preds.flatten()
        audio_pred = np.median(audio_preds)
        return audio_pred

    @staticmethod
    def fetch_percentile(z, values):
        values = sorted(values)

        for k, value in enumerate(values):
            if z < value:
                progress = k / len(values)
                return progress

        return 1

    @classmethod
    def get_percentiles(cls, preds, thresholds):
        all_percentiles = []
        for threshold in thresholds:
            percentile = cls.fetch_percentile(threshold, preds)
            all_percentiles.append(percentile)

        return all_percentiles

    def predict_faces(self, face_image_map):
        self.face_predict_timer.start()
        transform = self.face_predictor.transform
        per_face_pred = []

        for face_no in face_image_map:
            torch_images = []
            face_images = face_image_map.sample_detected_frames(
                face_no, max_samples=32
            )

            for frame_no in face_images:
                face = face_images[frame_no]
                face_image = face.image

                pil_image = Image.fromarray(face_image)
                torch_image = transform(pil_image)
                torch_image = torch.unsqueeze(torch_image, 0)
                torch_images.append(torch_image)

            torch_batch = torch.cat(torch_images, 0)
            preds = self.face_predictor.batch_predict(
                torch_batch, no_grad=True
            )

            face_pred = np.percentile(sorted(preds), 25)
            print(f'Q1 FACE PRED {face_pred}')
            per_face_pred.append(face_pred)

        if len(per_face_pred) != 0:
            face_pred = max(per_face_pred)
        else:
            print(f'FACELESS')
            face_pred = 0.85

        self.face_predict_timer.pause()
        return face_pred

    def handle_sync_predict(
        self, filename, face_image_map: FaceImageMap, audio_array,
        samples_holder: FaceSamplesHolder
    ):
        if len(face_image_map) == 0:
            return False

        # num_faces = len(face_image_map)
        cct = self.sync_predictor.load_cct(audio_array)

        for face_no in face_image_map:
            face_samples = face_image_map.sample_face_frames(
                face_no, consecutive_frames=5, extract=False,
                max_samples=32
            )
            samples_holder.add_face_samples(
                filename, face_samples=face_samples, mel=cct,
                face_no=face_no, fps=face_image_map.fps
            )

        return True

    @staticmethod
    def stderr(message):
        print(message, file=sys.stderr)

    @staticmethod
    def collate_sync_preds(sync_video_preds: dict):
        sync_face_preds = []

        for face_no in sync_video_preds:
            predictions = sync_video_preds[face_no]
            sync_face_pred = np.median(predictions)
            sync_face_preds.append(sync_face_pred)

        video_sync_pred = min(sync_face_preds)
        return video_sync_pred

    def main(self, input_dir, output_file, temp_dir=None):
        stderr = self.stderr
        output_dir = output_file[:output_file.rindex('/')]
        # input_dir = input_dir[:input_dir.rindex('/')]
        print(f'output dir {output_dir}')

        if temp_dir is None:
            temp_dir = f'{output_dir}/temp'
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        self.preds_holder = PredictionsHolder(input_dir, output_file)
        samples_holder = FaceSamplesHolder(
            predictor=self.sync_predictor,
            rgb_to_bgr=self.sync_rgb_swap, garbage_collect_cct=False,
            batch_size=self.face_batch_size,
            timer=self.sync_predict_timer,
            use_mouth_image=self.use_mouth_image
        )

        # read input directory for mp4 videos only
        # note: all files would be mp4 videos in the mounted input dir
        print(f'INPUT DIR {input_dir}')
        test_videos = self.get_test_videos(input_dir)
        random.shuffle(test_videos)

        stderr(f'TOTAL TEST VIDEOS: {len(test_videos)}')
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
            use_cuda=self.dataset_use_cuda or BIG_GPU,
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
        num_faceless_videos = 0
        num_videos = len(test_videos)
        self.timer.start()

        for sample in data_loader:
            if sample is None:
                print('END SAMPLE')
                continue

            filename, audio_array, face_image_map = sample
            audio_pred = self.predict_audio(audio_array)
            face_pred = self.predict_faces(face_image_map)

            has_sync_samples = self.handle_sync_predict(
                filename, face_image_map, audio_array,
                samples_holder=samples_holder
            )

            vid_pred_holder = self.preds_holder[filename]
            vid_pred_holder.face_pred = face_pred
            vid_pred_holder.audio_pred = audio_pred

            if not has_sync_samples:
                vid_pred_holder.sync_pred = 0.5
                num_faceless_videos += 1

        samples_holder.flush()
        all_sync_preds = samples_holder.make_video_preds()
        sync_preds_map, sync_confs_map = all_sync_preds

        for filename in sync_preds_map:
            vid_pred_holder = self.preds_holder[filename]
            sync_video_preds = sync_preds_map[filename]
            sync_video_confs = sync_confs_map[filename]
            sync_pred = self.collate_sync_preds(sync_video_preds)
            vid_pred_holder.sync_pred = sync_pred

        collate_pbar = tqdm(test_videos)
        # stderr(f'FACELESS VIDEOS = {num_faceless_videos}')
        print('TEST VIDEOS', test_videos)

        for filename in collate_pbar:
            vid_pred_holder = self.preds_holder[filename]

            face_pred = vid_pred_holder.face_pred
            audio_pred = vid_pred_holder.audio_pred
            sync_pred = vid_pred_holder.sync_pred
            video_pred = max(
                face_pred, audio_pred, sync_pred - 0.05
            )

            self.preds_holder.add_pred(filename, video_pred)

            face_status = f'FP={face_pred:2f}'
            audio_status = f'AP={audio_pred:2f}'
            sync_status = f'SP={sync_pred:2f}'

            print(f'filename [{k+1}]: {filename}')
            stats = f'{face_status}, {audio_status}, {sync_status}'
            desc = f'[{k+1}/{num_videos}] [{filename}] {stats}'
            # desc = f'[{k + 1}/{num_videos}] [{filename}]'

            stderr(desc)
            collate_pbar.set_description(desc)
            collate_pbar.update()
            k += 1

        self.timer.pause()
        self.preds_holder.export()

        stderr(f'total predict time: {self.timer.total}')
        stderr(f'face predict time: {self.face_predict_timer.total}')
        stderr(f'sync predict time: {self.sync_predict_timer.total}')
        stderr(f'audio predict time: {self.audio_predict_timer.total}')

        mem_allocated = torch.cuda.max_memory_allocated()
        mb_allocated = mem_allocated / (1024 ** 2)
        stderr(f'CUDA memory allocated: {mb_allocated} MB')

        all_preds = self.preds_holder.export_all_preds()
        face_preds, audio_preds, sync_preds = all_preds
        thresholds = [k / 10 for k in range(1, 10)]

        face_cdfs = self.get_percentiles(face_preds, thresholds)
        audio_cdfs = self.get_percentiles(audio_preds, thresholds)
        sync_cdfs = self.get_percentiles(sync_preds, thresholds)

        print('thresholds:', thresholds)
        print('face percentiles:', face_cdfs)
        print('audio percentiles:', audio_cdfs)
        print('sync percentiles:', sync_cdfs)


if __name__ == "__main__":
    mp.freeze_support()