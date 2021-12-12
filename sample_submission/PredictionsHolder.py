import os.path

import misc
import pandas as pd
import re

class VideoPrediction(object):
    def __init__(self, filename):
        self._filename = filename
        self._face_pred = None
        self._audio_pred = None
        self._sync_pred = None
        
        # self.default_value = None

    @property
    def filename(self):
        return self._filename

    @property
    def face_pred(self):
        return self._face_pred

    @face_pred.setter
    def face_pred(self, pred):
        assert self._face_pred is None
        self._face_pred = pred

    @property
    def audio_pred(self):
        return self._audio_pred

    @audio_pred.setter
    def audio_pred(self, pred):
        assert self._audio_pred is None
        self._audio_pred = pred

    @property
    def sync_pred(self):
        return self._sync_pred

    @sync_pred.setter
    def sync_pred(self, pred):
        assert self._sync_pred is None
        self._sync_pred = pred

    @property
    def max_pred(self):
        return max(
            self.face_pred, self.audio_pred. self.sync_pred
        )

    def show_stats(self, tag=None):
        print(f'PREDICTING [{tag}] [{self.filename}]')
        print(f'AUD-PRED = {self.audio_pred}')
        print(f'FACE-PRED = {self.face_pred}')
        print(f'SYNC-PRED = {self.sync_pred}')


class PredictionsHolder(object):
    def __init__(self, input_dir, output_file):
        if input_dir.endswith('/'):
            input_dir = input_dir[:-1]

        self.input_dir = input_dir
        self.output_file = output_file
        self.predictions = {}

        self.filenames = []
        self.probs = []

    def add_pred(self, filename, pred):
        self.filenames.append(filename)
        self.probs.append(pred)

    def __iter__(self):
        for filename in self.predictions:
            assert type(filename) is str
            yield filename

    def __getitem__(self, filename) -> VideoPrediction:
        return self.get(filename)

    def get(self, filename) -> VideoPrediction:
        match = re.match('^[a-f0-9]+\\.mp4$', filename)
        assert match is not None

        if filename not in self.predictions:
            try:
                assert filename.endswith('.mp4')
            except AssertionError as e:
                raise ValueError(f'BAD FILENAME {filename}')

            prediction_holder = VideoPrediction(filename)
            self.predictions[filename] = prediction_holder

        prediction_holder = self.predictions[filename]
        return prediction_holder

    def export_all_preds(self):
        face_preds, audio_preds, sync_preds = [], [], []

        for filename in self.filenames:
            vid_pred_holder = self.get(filename)

            face_pred = vid_pred_holder.face_pred
            audio_pred = vid_pred_holder.audio_pred
            sync_pred = vid_pred_holder.sync_pred

            face_preds.append(face_pred)
            audio_preds.append(audio_pred)
            sync_preds.append(sync_pred)

        output_df = pd.DataFrame({
            'filename': self.filenames,
            'face_pred': face_preds, 'audio_pred': audio_preds,
            'sync_pred': sync_preds
        })

        dirname = os.path.dirname(self.output_file)
        basename = os.path.basename(self.output_file)
        export_file = f'{dirname}/debug-{basename}'
        output_df.to_csv(export_file, index=False)
        print(f'exported to {export_file}')

        return face_preds, audio_preds, sync_preds

    def export(self):
        print(f'all exported filenames: {self.filenames}')
        print(f'all exported probs: {self.probs}')
        print(f'cleared: {len(self.filenames)}')

        # create output
        output_df = pd.DataFrame({
            "filename": self.filenames, "probability": self.probs
        })

        # write output as csv
        output_df.to_csv(self.output_file, index=False)
        print(f'exported to {self.output_file}')