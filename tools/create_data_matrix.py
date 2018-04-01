
from tqdm import tqdm
import os
from glob import glob
import cv2
import numpy as np
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--num_of_sample_per_file', type=int, default=500)
parser.add_argument('--num_class', type=int, default=2)
parser.add_argument('--synthetic_dir', type=str,
                    default='data/synthetic_data_numpy')
parser.add_argument('--num_of_sample', type=int,
                    default=10000)


arg = parser.parse_args()

def normalize(img):
    img = img/255.
    img = img*2-1
    return img


def to_matrix(b):
    rv = np.zeros(shape=[*b.shape[:2], 2])
    rv[:, :, 0] = b[:, :, 0]>0 
    rv[:, :, 1] = b[:, :, 2]>0
    return rv


def crop_input_output(img, is_normalize=False):
    # img = cv2.imread(path)
    if is_normalize:
        img = normalize(img)
    w_full = img.shape[1]
    img = cv2.resize(img, (2048, img.shape[0]))
    a = img[:, :w_full//2]
    b = img[:, w_full//2:]
    h, w = a.shape[:2]
    sx = np.random.choice(w-1024) if w > 1024 else 0
    if h > 128:
        sy = np.random.choice(h-128)
    else:
        sy = 0

    a = a[sy:sy+128, sx:sx+1024]
    b = b[sy:sy+128, sx:sx+1024]
    a = cv2.resize(a, (1024, 128))
    b = cv2.resize(b, (1024, 128))
    a = normalize(a)
    b = b / 255.0
    b = to_matrix(b)
    return a, b


if __name__ == '__main__':
    paths = glob('images/train/*.png')
    img_roots = [cv2.imread(path) for path in paths]
    os.makedirs(arg.synthetic_dir, exist_ok=True)
    inputs, labels = [], []
    for i in tqdm(range(arg.num_of_sample)):
        choice = np.random.choice(len(img_roots))
        img_goc = img_roots[choice]        
        scale_size = np.random.uniform(300/1024, 1900/1024)
        img = cv2.resize(img_goc, (2048, int(scale_size*img_goc.shape[0])))
        a, b = crop_input_output(img)
        inputs.append(a)
        labels.append(b)
        if len(labels) >= arg.num_of_sample_per_file:
            # ims = {'inputs': np.array(inputs), 'labels': np.array(labels)}
            save_path = '{}/{}'.format(arg.synthetic_dir, i)
            np.savez_compressed(save_path, inputs=inputs, labels=labels)
            del inputs, labels
            inputs, labels = [], []
