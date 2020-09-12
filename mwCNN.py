import tensorflow as tf
from tensorflow.keras.layers import Layer
class DWT(Layer):
    def call(self, inputs):
        # taken from
        # https://github.com/lpj-github-io/MWCNNv2/blob/master/MWCNN_code/model/common.py#L65
        x01 = inputs[:, 0::2] / 2
        x02 = inputs[:, 1::2] / 2
        x1 = x01[:, :, 0::2]
        x2 = x02[:, :, 0::2]
        x3 = x01[:, :, 1::2]
        x4 = x02[:, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return tf.concat((x_LL, x_HL, x_LH, x_HH), axis=-1)
