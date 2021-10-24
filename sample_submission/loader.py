import cv2
import numpy as np
import pandas as pd

class BoundingBox(object):
    def __init__(self, x_start, x_end, y_start, y_end):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end

    def to_tuple(self):
        return (
            self.x_start, self.x_end,
            self.y_start, self.y_end
        )

    def __index__(self, k):
        return self.to_tuple()[k]

    def __repr__(self):
        name = self.__class__.__name__
        args = ', '.join([
            repr(arg) for arg in self.to_tuple()
        ])

        return f'{name}({args})'

    @staticmethod
    def rescale_coord(coord, new_start, new_end, end):
        new_coord = end * (coord - new_start) / (new_end - new_start)
        new_coord = int(max(min(new_coord, end), 0))
        return new_coord

    def rescale_x(self, x, width):
        return self.rescale_coord(
            x, self.x_start, self.x_end, width
        )

    def rescale_y(self, y, height):
        return self.rescale_coord(
            y, self.y_start, self.y_end, height
        )

    def rescale(self, x, y, image):
        height = image.shape[0]
        width = image.shape[1]

        new_x = self.rescale_coord(x, self.x_start, self.x_end, width)
        new_y = self.rescale_coord(y, self.y_start, self.y_end, height)
        return new_x, new_y


class VideoArray(object):
    def __init__(
        self, out_video, width, height, frames,
        name, out_pil=None, rescale=1, frame_mapping=None,
        raw_video=None
    ):
        self.name = name
        self.out_video = out_video
        self.width = width
        self.height = height
        self.frames = frames

        self.frame_mapping = frame_mapping

        self.out_pil = out_pil
        self.rescale = rescale
        self.raw_video = raw_video

    def __repr__(self):
        class_name = self.__class__.__name__
        kwargs = ', '.join([
            f'name={repr(self.name)}',
            f'width={self.width}',
            f'height={self.height}'
        ])
        return f'{class_name}({kwargs})'

    def get_grayscale_frames(self):
        gray_frames = []

        for frame in self.out_video:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frames.append(gray)

        return gray_frames

    def get_face_images(self, video_faces, every_n_frames=20):
        frame_indexes = list(video_faces.keys())
        face_images = []

        for frame_index in frame_indexes:
            video_index = frame_index // every_n_frames
            frame = self.out_video[video_index]
            locations = video_faces[frame_index]

            for face_location in locations:
                top, right, bottom, left = face_location
                crop_image = frame[top:bottom, left:right]
                face_images.append(crop_image)

        return face_images

    def get_rescale_ratios(self, *args, **kwargs):
        coords = self.cut_blackout(self.out_video, *args, **kwargs)
        x_start, x_end, y_start, y_end = coords.to_tuple()

        x_scale = self.width / (x_end - x_start)
        y_scale = self.height / (y_end - y_start)
        return x_scale, y_scale

    def auto_resize(self, *args, **kwargs):
        resolution = (self.width, self.height)
        coords = self.cut_blackout(self.out_video, *args, **kwargs)
        x_start, x_end, y_start, y_end = coords.to_tuple()
        # print(f'COORDS {coords}')
        resized_frames = []

        for frame in self.out_video:
            cropped_frame = frame[y_start:y_end, x_start:x_end]
            resized_frame = cv2.resize(cropped_frame, resolution)
            # print('F-SHAPE', resized_frame.shape)
            expanded_frame = np.expand_dims(resized_frame, axis=0)
            resized_frames.append(expanded_frame)

        resized_frames = np.concatenate(resized_frames)

        return VideoArray(
            resized_frames, width=self.width, height=self.height,
            frames=self.frames, rescale=self.rescale, name=self.name
        )

    def cut_blackout(
        self, images=None, samples=1, intervals=5, roll=4
    ):
        if images is None:
            images = self.out_video

        sample_interval = len(images) // samples
        x_starts, y_starts = [], []
        x_ends, y_ends = [], []

        for k in range(samples):
            interval = sample_interval * k
            # print(f'INTERVAL {interval} {images.shape}')
            frame = images[interval]

            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                assert len(frame.shape) == 2
                gray_frame = frame

            _, binary_frame = cv2.threshold(
                gray_frame, 1, 255, cv2.THRESH_BINARY
            )

            x_start, x_end = self.horizontal_clip(
                binary_frame, intervals=intervals, roll=roll
            )
            y_start, y_end = self.vertical_clip(
                binary_frame, intervals=intervals, roll=roll
            )

            x_starts.append(x_start)
            x_ends.append(x_end)
            y_starts.append(y_start)
            y_ends.append(y_end)

        x_start = int(np.median(x_starts))
        x_end = int(np.median(x_ends))
        y_start = int(np.median(y_starts))
        y_end = int(np.median(y_ends))

        return BoundingBox(x_start, x_end, y_start, y_end)

    @classmethod
    def h_clip(cls, *args, **kwargs):
        return cls.horizontal_clip(*args, **kwargs)

    @classmethod
    def v_clip(cls, *args, **kwargs):
        return cls.vertical_clip(*args, **kwargs)

    @classmethod
    def horizontal_clip(cls, bin_image, intervals=5, roll=5):
        first_indexes, last_indexes = [], []
        interval_length = len(bin_image) // intervals

        for k in range(intervals):
            interval = interval_length * k
            h_strip = bin_image[interval]

            blackout = cls.get_strip_blackout(h_strip, roll=roll)
            start_index, end_index = blackout
            first_indexes.append(start_index)
            last_indexes.append(end_index)

        clip_start = int(np.median(first_indexes))
        clip_end = int(np.median(last_indexes))
        return clip_start, clip_end

    @classmethod
    def vertical_clip(cls, bin_image, intervals=5, roll=5):
        first_indexes, last_indexes = [], []
        interval_length = len(bin_image[0]) // intervals

        for k in range(intervals):
            interval = interval_length * k
            v_strip = bin_image[:, interval]

            blackout = cls.get_strip_blackout(v_strip, roll=roll)
            start_index, end_index = blackout
            first_indexes.append(start_index)
            last_indexes.append(end_index)

        clip_start = int(np.median(first_indexes))
        clip_end = int(np.median(last_indexes))
        return clip_start, clip_end

    @staticmethod
    def get_strip_blackout(img_strip, roll=5):
        pool_strip = pd.DataFrame(img_strip).rolling(roll).median()
        pool_strip = pool_strip.to_numpy().ravel()
        clip_strip = pool_strip[roll - 1:]
        # print('CLIP STRIP', clip_strip)

        indexes = np.where(clip_strip != 0)[0]

        try:
            last_normal_index = max(indexes)
            # print(indexes, last_normal_index)
            last_normal_index += roll - 1
        except ValueError:
            last_normal_index = len(img_strip)

        if pool_strip[roll] == 0:
            indexes = np.where(clip_strip != 0)[0]

            try:
                first_normal_index = min(indexes)
                # print(indexes, first_normal_index)
                first_normal_index += roll // 2 - 1
            except ValueError:
                first_normal_index = 0
        else:
            first_normal_index = 0

        return first_normal_index, last_normal_index


def load_video(
    cap, every_n_frames=None, specific_frames=None,
    to_rgb=True, scale=1, inc_pil=False,
    max_frames=None, release=True, filename=None,
    verbose=False, early_stop=True
):
    """
    Loads a video.
    Called by:
    1) The finding faces algorithm where it pulls a frame every
    FACE_FRAMES frames up to MAX_FRAMES_TO_LOAD at a scale of
    FACEDETECTION_DOWNSAMPLE, and then half that if there's a
    CUDA memory error.
    2) The inference loop where it pulls EVERY frame up to a certain
    amount which it the last needed frame for each face for that video
    """

    assert every_n_frames or specific_frames, (
        "Must supply either every n_frames or specific_frames"
    )
    assert bool(every_n_frames) != bool(specific_frames), (
        "Supply either 'every_n_frames' or 'specific_frames', not both"
    )

    if type(cap) == str:
        if filename is None:
            filename = cap

        cap = cv2.VideoCapture(filename)

    n_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f'NUM FRAMES {n_frames_in}')

    if n_frames_in == 0:
        return None

    width_out = int(width_in * scale) if scale else width_in
    height_out = int(height_in * scale) if scale else height_in

    if verbose:
        print(f'ORIGINAL RES', (width_in, height_in))
        print(f'NEW RES', (width_out, height_out), scale)

    if max_frames:
        n_frames_in = min(n_frames_in, max_frames)

    if every_n_frames:
        specific_frames = list(range(0, n_frames_in, every_n_frames))

    n_frames_out = len(specific_frames)
    frame_mapping = {}
    out_pil = []

    out_video = np.empty(
        (n_frames_out, height_out, width_out, 3),
        np.dtype('uint8')
    )

    i_frame_in = 0
    i_frame_out = 0
    frame_no = 0
    ret = True

    last_needed_frame = max(specific_frames)

    while i_frame_in < n_frames_in and ret:
        if early_stop and (i_frame_in > last_needed_frame):
            break

        try:
            try:
                if every_n_frames == 1:
                    # Faster if reading all frames
                    ret, frame_in = cap.read()
                else:
                    ret = cap.grab()

                    if i_frame_in not in specific_frames:
                        i_frame_in += 1
                        continue

                    ret, frame_in = cap.retrieve()

                if scale:
                    frame_in = cv2.resize(
                        frame_in, (width_out, height_out)
                    )
                if to_rgb:
                    frame_in = cv2.cvtColor(
                        frame_in, cv2.COLOR_BGR2RGB
                    )

            except Exception as e:
                print(
                    f"Error for frame {i_frame_in} for video" +
                    f" {filename}: {e}; using 0s"
                )
                frame_in = np.zeros((height_out, width_out, 3))

            out_video[i_frame_out] = frame_in
            frame_mapping[i_frame_in] = frame_in
            i_frame_out += 1

            if inc_pil:
                try:
                    # https://www.kaggle.com/zaharch/public-test-errors
                    pil_img = Image.fromarray(frame_in)
                except Exception as e:
                    print(
                        f"Using a blank frame for video {filename}" +
                        f" frame {i_frame_in} as error {e}"
                    )
                    # Use a blank frame
                    pil_img = Image.fromarray(
                        np.zeros((224, 224, 3), dtype=np.uint8)
                    )

                out_pil.append(pil_img)

            i_frame_in += 1

        except Exception as e:
            print(f"Error for file {filename}: {e}")

    if release:
        cap.release()

    return VideoArray(
        out_video, width=width_out, height=height_out,
        frames=n_frames_out, out_pil=out_pil, rescale=scale,
        raw_video=cap, name=filename, frame_mapping=frame_mapping
    )


if __name__ == '__main__':
    path = '../datasets/train/tmc_train_00/0ae95b34e9481b4f.mp4'
    output = load_video(path, every_n_frames=10)
    video = output[0]

    print(f'OUT TYPE: {type(video)}')

    width = video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    print(f'RES: {width} {height}')