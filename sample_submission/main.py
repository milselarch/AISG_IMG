import time
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
from AISG.ParallelFaceExtract import ParallelFaceExtract

from Timer import Timer
from PredictionsHolder import PredictionsHolder

vram_gb = misc.get_gpu_capacity()
BIG_GPU = True if vram_gb > 12 else False
print(f'GPU VRAM = {vram_gb}GB, USE-GPU [{BIG_GPU}]')
print('VERSION 0.0.8')

face_all_timer = Timer()
face_predict_timer = Timer()

audio_predictor = AudioPredictor(
    preload_path='models/AUD-211002-1735.pt',
    cache_threshold=20, load_dataset=False, use_batch_norm=True,
    add_aisg=False, use_avs=True, train_version=1,
    use_cuda=BIG_GPU
)
face_predictor = MesoTrainer(
    preload_path='models/MES-211022-0001.pt',
    load_dataset=False, use_cuda=BIG_GPU
)

# print(os.system('ls -a'))
# print(os.system('ls /data/input -a'))
# input('TEST ')

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

def handle_face_preds(holder, extractor):
    transform = face_predictor.transform

    while True:
        result = extractor.pop()

        if result is None:
            print('RESULT IS NONE')
            return

        filepath, face_image_map = result
        print('MAP FILEPATH =', filepath)
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
                torch_images.append(torch_image)

            face_predict_timer.start()
            preds = face_predictor.predict_images(torch_images)
            face_predict_timer.pause()

            face_pred = np.percentile(sorted(preds), 75)
            print(f'F-PRED {face_pred}')
            per_face_pred.append(face_pred)

        if len(per_face_pred) != 0:
            face_pred = max(per_face_pred)
        else:
            print(f'FACELESS {filename}')
            face_pred = 0.85

        print(f'ADD POP RESULT', filepath)
        print(f'FACE PRED [{name}] = {face_pred}')
        holder.add_face_pred(filename, face_pred)

def main(input_dir, output_file, temp_dir=None):
    output_dir = output_file[:output_file.rindex('/')]
    print(f'output dir {output_dir}')

    if temp_dir is None:
        temp_dir = f'{output_dir}/temp'

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

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

    face_extractor = ParallelFaceExtract()
    face_extractor.start(filepaths=test_videos, base_dir=input_dir)
    extract_audios(test_videos, input_dir, temp_dir)

    preds_holder = PredictionsHolder(input_dir, output_file)
    pbar = tqdm(test_videos)

    for filename in pbar:
        name = misc.path_to_name(filename)

        audio_filepath = f'{temp_dir}/{name}.flac'
        audio_preds = audio_predictor.batch_predict(audio_filepath)
        audio_preds = audio_preds.flatten()
        audio_pred = np.median(audio_preds)
        preds_holder.add_audio_pred(filename, audio_pred)
        print(f'AUD PRED [{name}] = {audio_pred}')

        with face_all_timer:
            handle_face_preds(preds_holder, face_extractor)

    # video_filepath = f'{input_dir}/{filename}'
    print(f'WAITING ON FACE EXTRACTOR')
    print(f'EXTRACTOR SIZE {face_extractor.size}')

    with face_all_timer:
        while not face_extractor.is_done:
            handle_face_preds(preds_holder, face_extractor)
            time.sleep(1)

    print(f'face time spent: {face_all_timer.total}')
    print(f'face predict time spent: {face_predict_timer.total}')
    preds_holder.export()


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
