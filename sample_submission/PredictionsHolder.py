import misc

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
        name = misc.path_to_name(filepath)
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

        print(f'U-PRED [{filename}] F={face_pred} A={audio_pred}')
        print(f'uncleared faces: {len(self.face_preds)}')
        print(f'uncleared audios: {len(self.audio_preds)}')
        print(f'cleared: {len(self.filenames)}')

        self.filenames.append(filename)
        self.probs.append(overall_pred)
        return True

    def add_face_pred(self, filepath, pred):
        print(f'U-ADD FACE PRED {filepath} {pred}')
        self.face_preds[filepath] = pred
        self.update(filepath)

    def add_audio_pred(self, filepath, pred):
        print(f'U-ADD AUDIO PRED {filepath} {pred}')
        self.audio_preds[filepath] = pred
        self.update(filepath)

    def export(self):
        print(f'uncleared faces: {self.face_preds}')
        print(f'uncleared audios: {self.audio_preds}')
        print(f'all exported filenames: {self.filenames}')
        print(f'all exported probs: {self.probs}')
        print(f'cleared: {len(self.filenames)}')

        try:
            assert len(self.audio_preds) == 0
            assert len(self.face_preds) == 0
        except AssertionError as e:
            print('NOT ALL VIDEOS CLEARED')
            raise e

        # create output
        output_df = pd.DataFrame({
            "filename": self.filenames, "probability": self.probs
        })

        # write output as csv
        output_df.to_csv(self.output_file, index=False)
        print(f'exported to {self.output_file}')