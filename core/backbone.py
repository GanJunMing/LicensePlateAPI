import core.common as common

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data


# def cspdarknet53(input_data):

#     input_data = common.convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
#     input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

#     route = input_data
#     route = common.convolutional(route, (1, 1, 64, 64), activate_type="mish")
#     input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
#     for i in range(1):
#         input_data = common.residual_block(input_data,  64,  32, 64, activate_type="mish")
#     input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

#     input_data = tf.concat([input_data, route], axis=-1)
#     input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
#     input_data = common.convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
#     route = input_data
#     route = common.convolutional(route, (1, 1, 128, 64), activate_type="mish")
#     input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
#     for i in range(2):
#         input_data = common.residual_block(input_data, 64,  64, 64, activate_type="mish")
#     input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
#     input_data = tf.concat([input_data, route], axis=-1)

#     input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
#     input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
#     route = input_data
#     route = common.convolutional(route, (1, 1, 256, 128), activate_type="mish")
#     input_data = common.convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
#     for i in range(8):
#         input_data = common.residual_block(input_data, 128, 128, 128, activate_type="mish")
#     input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
#     input_data = tf.concat([input_data, route], axis=-1)

#     input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
#     route_1 = input_data
#     input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
#     route = input_data
#     route = common.convolutional(route, (1, 1, 512, 256), activate_type="mish")
#     input_data = common.convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
#     for i in range(8):
#         input_data = common.residual_block(input_data, 256, 256, 256, activate_type="mish")
#     input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
#     input_data = tf.concat([input_data, route], axis=-1)

#     input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
#     route_2 = input_data
#     input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
#     route = input_data
#     route = common.convolutional(route, (1, 1, 1024, 512), activate_type="mish")
#     input_data = common.convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
#     for i in range(4):
#         input_data = common.residual_block(input_data, 512, 512, 512, activate_type="mish")
#     input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
#     input_data = tf.concat([input_data, route], axis=-1)

#     input_data = common.convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
#     input_data = common.convolutional(input_data, (1, 1, 1024, 512))
#     input_data = common.convolutional(input_data, (3, 3, 512, 1024))
#     input_data = common.convolutional(input_data, (1, 1, 1024, 512))

#     input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
#                             , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
#     input_data = common.convolutional(input_data, (1, 1, 2048, 512))
#     input_data = common.convolutional(input_data, (3, 3, 512, 1024))
#     input_data = common.convolutional(input_data, (1, 1, 1024, 512))

#     return route_1, route_2, input_data

