import tensorflow as tf


def extract_patches(image, k_size, strides):
    images = tf.extract_image_patches(tf.expand_dims(
        image, 0), k_size, strides, rates=[1, 1, 1, 1], padding='SAME')[0]
    images_shape = tf.shape(images)
    images_reshape = tf.reshape(
        images, [images_shape[0]*images_shape[1], *k_size[1:3], 3])
    images, n1, n2 = tf.cast(images_reshape, tf.uint8) , images_shape[0], images_shape[1]
    return images, n1, n2

def join_patches(images, n1, n2, k_size, strides):

    s1 = k_size[1]//2-strides[1]//2
    s2 = k_size[2]//2-strides[2]//2
    roi = images[:, 
                 s1:s1+strides[1],\
                 s2:s2+strides[2],
                 :]
    new_shape = [n1, n2, *roi.get_shape().as_list()[1:]]
    reshaped_roi = tf.reshape(roi, new_shape)
    reshaped_roi = tf.transpose(reshaped_roi, perm=[0,2,1,3,4])
    rs = tf.shape(reshaped_roi)
    rv = tf.cast(tf.reshape(reshaped_roi, [rs[0]*rs[1], rs[2]*rs[3], -1]), tf.uint8)
    return rv