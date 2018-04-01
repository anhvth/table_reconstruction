import os
from glob import glob
import cv2
import numpy as np
import os
import argparse
import _thread
import time
parser = argparse.ArgumentParser()

parser.add_argument('--n_sample', type=int, default=100,help='number of sample per images')

parser = parser.parse_args()

def normalize(img):
    img = img/255
    img = img*2-1
    return img


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
    return a, b


def gen_image(path, thread=None):
    
    img_goc = cv2.imread(path)
    info = path
    if thread is not None:
        info += thread
    print(info)
    synthetic_dir = 'data/synthetic_data_bounding'
    os.makedirs(synthetic_dir, exist_ok=True)
    os.makedirs(synthetic_dir, exist_ok=True)
    name = os.path.split(path)[-1].split('.')[0]
    for i in range(parser.n_sample):
        scale_size = np.random.uniform(0.7, 1.3)
        img = cv2.resize(img_goc, (2048, int(scale_size*img_goc.shape[0])))
        a, b = crop_input_output(img)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        a = np.stack([a]*3, axis=2)
        im = np.concatenate([a, b], axis=1)
        save_path = '{}/{}_{}.png'.format(synthetic_dir, name, i)
        # print(save_path)
        cv2.imwrite(save_path, im)


if __name__ == '__main__':
    paths = sorted(glob('data/train_bounding/*.png'))
    print('Num of inputs: ', len(paths))
    for i in range(0, len(paths), 1):
        gen_image(paths[i])
        # try:
        #     _thread.start_new_thread( gen_image, (paths[i], "first", ) )
        #     _thread.start_new_thread( gen_image, (paths[i+1], "second", ) )
        # except:
        #     print ("Error: unable to start thread")
        # while 1:
        #     pass