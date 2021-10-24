import time

import pandas as pd
import subprocess
import numpy as np
import nvidia_smi
import os

from AISG.loader import load_video
from AISG.DeepfakeDetection.FaceExtractor import FaceExtractor
from AISG.MesoNet.MesoTrainer import MesoTrainer
from AISG.FakeVoiceTorch.trainer import Trainer as AudioPredictor
from AISG.ParallelFaceExtract import ParallelFaceExtract

from Timer import Timer
from argparse import ArgumentParser
from numpy import random
from tqdm.auto import tqdm
from PIL import Image

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
vram_gb = gpu_info.total / 1e9

if vram_gb < 12:
    use_gpu = False
else:
    use_gpu = True

print(f'GPU VRAM = {vram_gb}GB, USE-GPU [{use_gpu}]')
face_all_timer = Timer()
face_predict_timer = Timer()

audio_predictor = AudioPredictor(
    preload_path='models/AUD-211002-1735.pt',
    cache_threshold=20, load_dataset=False, use_batch_norm=True,
    add_aisg=False, use_avs=True, train_version=1,
    use_cuda=use_gpu
)
face_predictor = MesoTrainer(
    preload_path='models/MES-211022-0001.pt',
    load_dataset=False, use_cuda=use_gpu
)

# print(os.system('ls -a'))
# print(os.system('ls /data/input -a'))
# input('TEST ')

def extract_audios(test_videos, input_dir, out_dir):
    pbar = tqdm(test_videos)

    for filename in pbar:
        pbar.set_description(f'extracting {filename}')
        filepath = f'{input_dir}{filename}'
        name = filename[:filename.index('.')]

        ffmpeg_args = '-ab 160k -ac 2 -ar 44100'
        out_path = f'{out_dir}/{name}.flac'
        subprocess.run(
            f'ffmpeg -i {filepath} {ffmpeg_args} -vn {out_path}',
            capture_output=True, shell=True
        )

def export_face_images(name, face_image_map):
    export_dir = '/data/output/images'

    for face_no in face_image_map:
        face_images = face_image_map[face_no]
        face_dir = f'{export_dir}/{name}'
        if not os.path.exists(face_dir):
            os.mkdir(face_dir)

        for k, face_image in enumerate(face_images):
            im = Image.fromarray(face_image)
            path = f'{face_dir}/{face_no}-{k}.jpg'
            im.save(path)

def path_to_name(filepath):
    name = filepath

    if '/' in name:
        name = name[name.rindex('/') + 1:]
    if '.' in name:
        name = name[:name.index('.')]

    return name

class PredictionsHolder(object):
    def __init__(self, input_dir, output_file):
        if input_dir.endswith('/'):
            input_dir = input_dir[:-1]

        self.input_dir = input_dir
        self.output_file = output_file
        self.audio_preds = {}
        self.face_preds = {}

        self.filenames = []
        self.probs = []

    def update(self, filepath):
        name = path_to_name(filepath)
        filename = f'{name}.mp4'

        # print(f'UPDATE {filename}')
        # print(f'FACE PREDS {self.face_preds}')
        # print(f'AUDIO PREDS {self.audio_preds}')

        if filename not in self.face_preds:
            return False
        if filename not in self.audio_preds:
            return False

        audio_pred = self.audio_preds[filename]
        face_pred = self.face_preds[filename]
        overall_pred = max(audio_pred, face_pred)
        del self.audio_preds[filename]
        del self.face_preds[filename]

        print(f'PRED [{filename}] F={face_pred} A={audio_pred}')
        self.filenames.append(filename)
        self.probs.append(overall_pred)
        return True

    def add_face_pred(self, filepath, pred):
        self.face_preds[filepath] = pred
        self.update(filepath)

    def add_audio_pred(self, filepath, pred):
        self.audio_preds[filepath] = pred
        self.update(filepath)

    def export(self):
        # create output
        assert len(self.audio_preds) == 0
        assert len(self.face_preds) == 0

        output_df = pd.DataFrame({
            "filename": self.filenames, "probability": self.probs
        })

        # write output as csv
        output_df.to_csv(self.output_file, index=False)
        print(f'exported to {self.output_file}')

def handle_face_preds(holder, extractor):
    while True:
        result = extractor.pop()
        if result is None:
            return

        transform = face_predictor.transform
        filepath, face_image_map = result
        # print(f'MAP FILEPATH = {filepath}')
        name = path_to_name(filepath)
        filename = f'{name}.mp4'
        per_face_pred = []

        for face_no in face_image_map:
            face_images = face_image_map[face_no]
            torch_images = [
                transform(Image.fromarray(face_image))
                for face_image in face_images
            ]

            with face_predict_timer:
                preds = face_predictor.predict_images(torch_images)

            face_pred = np.percentile(sorted(preds), 75)
            print(f'F-PRED {face_pred}')
            per_face_pred.append(face_pred)

        if len(per_face_pred) != 0:
            face_pred = max(per_face_pred)
        else:
            print(f'FACELESS {filename}')
            face_pred = 0.85

        print(f'FACE PRED [{name}] = {face_pred}')
        holder.add_face_pred(filename, face_pred)

def main(input_dir, output_file):
    # read input directory for mp4 videos only
    # note: all files would be mp4 videos in the mounted input dir
    print(f'INPUT DIR {input_dir}')
    test_videos = [
        video for video in os.listdir(input_dir)
        if ".mp4" in video
    ]

    temp_dir = 'temp'
    face_extractor = ParallelFaceExtract()
    face_extractor.start(filepaths=test_videos, base_dir=input_dir)
    extract_audios(test_videos, input_dir, temp_dir)

    preds_holder = PredictionsHolder(input_dir, output_file)
    pbar = tqdm(test_videos)

    for filename in pbar:
        name = path_to_name(filename)

        audio_filepath = f'{temp_dir}/{name}.flac'
        audio_preds = audio_predictor.batch_predict(audio_filepath)
        audio_preds = audio_preds.flatten()
        audio_pred = np.median(audio_preds)
        preds_holder.add_audio_pred(filename, audio_pred)
        print(f'AUD PRED [{name}] = {audio_pred}')

        with face_all_timer:
            handle_face_preds(preds_holder, face_extractor)

    # video_filepath = f'{input_dir}/{filename}'
    with face_all_timer:
        while not face_extractor.is_done:
            handle_face_preds(preds_holder, face_extractor)
            time.sleep(0.1)

    preds_holder.export()
    print(f'face time spent: {face_all_timer.total}')
    print(f'face predict time spent: {face_predict_timer.total}')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-input", type=str, required=True,
        help="Input directory of test videos"
    )
    parser.add_argument(
        "-output", type=str, required=True,
        help="Output dir with filename e.g. /data/submission.csv"
    )

    args = parser.parse_args()
    main(input_dir=args.input, output_file=args.output)
