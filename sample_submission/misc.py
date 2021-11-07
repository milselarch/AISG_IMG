import nvidia_smi

from PIL import Image
from datetime import datetime

def kwargify(**kwargs):
    return kwargs

def get_gpu_capacity():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    vram_gb = gpu_info.total / 1e9
    return vram_gb

def make_date_stamp():
    return datetime.now().strftime("%y%m%d-%H%M")

def path_to_name(filepath):
    name = filepath

    if '/' in name:
        name = name[name.rindex('/') + 1:]
    if '.' in name:
        name = name[:name.index('.')]

    return name

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