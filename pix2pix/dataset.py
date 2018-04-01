import tensorflow as tf 
from glob import glob 
import numpy as np
def get_data_generator(input_dir, batch_size=3, n_repeat=1):
    paths = glob('{}/*'.format(input_dir))
    while True:
        path = paths[np.random.choice(len(paths))]
        with np.load(path) as data:
            inputs = data['inputs']
            labels = data['labels']
            idxs = np.arange(len(inputs))
            np.random.shuffle(idxs)
            inputs = inputs[idxs]
            labels = labels[idxs]
        x, y = [], []
        for i in range(0, len(inputs), batch_size):
            k = min(len(inputs), i+batch_size)
            # print('yield:', i, k)
            yield inputs[i:k], labels[i:k]

def labelmap_to_image(labelmap):
    h, w = labelmap.shape[:2]
    image = np.zeros(shape=[h,w,3])
    im1 = labelmap[:,:,:1] * np.array([255,0,0]).reshape([1,1,3])
    im2 = labelmap[:,:,2:3] * np.array([0,0,255]).reshape([1,1,3])
    im = im1+im2
    return np.clip(im.astype(np.uint8), a_min=0, a_max=255)

if __name__ == '__main__':
    dataset = get_data_generator('data/synthetic_data_numpy')
    a, b = next(dataset)
    lbl = b[0]
    img = labelmap_to_image(lbl)
    import cv2
    cv2.imwrite('test.png', img)