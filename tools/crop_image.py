import cv2 
from glob import glob 
import numpy as np 
import os 

def crop(img):
    w = img.shape[1]
    a = img[:, :w//2]
    b = img[:, w//2:]
    h, w = a.shape[:2]
    b = cv2.resize(b, (w, h))
    # print(a.shape, b.shape)
    sx = np.random.choice(w-256) if w > 256 else 0
    sy = np.random.choice(h-256) if h > 256 else 0
    def f(_):
        pad = 255*np.ones(shape=[256,256,3])
        _ = _[sy:sy+256, sx:sx+256] 
        # print(_)
        h, w = _.shape[:2]
        pad[:h, :w] = _
        return pad 
    output = np.concatenate([f(a), f(b)], axis=1)
    return output

if __name__ == '__main__':
    paths = glob('tabels/contour_pairs/*.jpg')
    os.makedirs('tabels/contour_combined/', exist_ok=True)
    names = [os.path.split(path)[-1] for path in paths]
    for path, name in zip(paths, names):
        img = cv2.imread(path)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=.5)
        for i in range(100):
            img_ = crop(img)
            cv2.imwrite('tabels/contour_combined/{}_{}'.format(i, name), img_)
            # print(img_.shape)