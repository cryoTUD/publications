######################################################## IMPORTS ##################################################################

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend, Model
from tensorflow.keras.layers import Input, Conv3D, Activation, MaxPooling3D, PReLU, ReLU, ELU, Dropout
from tensorflow.keras.layers import Conv3DTranspose, UpSampling3D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.regularizers import L1L2

######################################################## FUNCTIONS ################################################################

def define_model(cube_shape):
    """ defines a 3D CNN U-Net model with eight convolutional layers

    Args:
        cube_shape (int): cube shape in x, y, z direction

    Returns:
        UNet_model (tf.keras.Model): trainable model structure of 3D CNN U-Net
    """

    print("\n>>> DEFINE MODEL")

    inpt = Input(shape=np.zeros((cube_shape, cube_shape, cube_shape, 1)).shape)

    # encoder layer 1
    conv1_1_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv1_1_3d")(inpt)
    conv1_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv1_1_3d)
    conv1_1_GN = GroupNormalization(groups=8, axis=-1)(conv1_1_PReLU)

    conv1_2_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv1_2_3d")(conv1_1_GN)
    conv1_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv1_2_3d)
    conv1_2_GN = GroupNormalization(groups=8, axis=-1)(conv1_2_PReLU)

    conv1_3_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=2, name="conv1_3_3d")(conv1_2_GN)
    conv1_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv1_3_3d)
    conv1_3_GN = GroupNormalization(groups=8, axis=-1)(conv1_3_PReLU)

    # encoder layer 2
    conv2_1_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv2_1_3d")(conv1_3_GN)
    conv2_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv2_1_3d)
    conv2_1_GN = GroupNormalization(groups=8, axis=-1)(conv2_1_PReLU)

    conv2_2_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv2_2_3d")(conv2_1_GN)
    conv2_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv2_2_3d)
    conv2_2_GN = GroupNormalization(groups=8, axis=-1)(conv2_2_PReLU)

    conv2_3_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=2, name="conv2_3_3d")(conv2_2_GN)
    conv2_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv2_3_3d)
    conv2_3_GN = GroupNormalization(groups=8, axis=-1)(conv2_3_PReLU)

    # encoder layer 3
    conv3_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv3_1_3d")(conv2_3_GN)
    conv3_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv3_1_3d)
    conv3_1_GN = GroupNormalization(groups=8, axis=-1)(conv3_1_PReLU)

    conv3_2_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv3_2_3d")(conv3_1_GN)
    conv3_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv3_2_3d)
    conv3_2_GN = GroupNormalization(groups=8, axis=-1)(conv3_2_PReLU)

    conv3_3_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=2, name="conv3_3_3d")(conv3_2_GN)
    conv3_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv3_3_3d)
    conv3_3_GN = GroupNormalization(groups=8, axis=-1)(conv3_3_PReLU)

    # bottom layer
    conv4_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv4_1_3d")(conv3_3_GN)
    conv4_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv4_1_3d)
    conv4_1_GN = GroupNormalization(groups=8, axis=-1)(conv4_1_PReLU)

    # decoder layer 1
    up5 = Conv3DTranspose(filters=128, kernel_size=5, padding='same', strides=2)(conv4_1_GN)
    add5 = concatenate([up5, conv3_1_GN], axis=-1)

    conv5_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv5_1_3d")(add5)
    conv5_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv5_1_3d)
    conv5_1_GN = GroupNormalization(groups=8, axis=-1)(conv5_1_PReLU)

    conv5_2_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv5_2_3d")(conv5_1_GN)
    conv5_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv5_2_3d)
    conv5_2_GN = GroupNormalization(groups=8, axis=-1)(conv5_2_PReLU)

    conv5_3_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv5_3_3d")(conv5_2_GN)
    conv5_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv5_3_3d)
    conv5_3_GN = GroupNormalization(groups=8, axis=-1)(conv5_3_PReLU)

    # decoder layer 2
    up6 = Conv3DTranspose(filters=64, kernel_size=5, padding='same', strides=2)(conv5_3_GN)
    add6 = concatenate([up6, conv2_1_GN], axis=-1)

    conv6_1_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv6_1_3d")(add6)
    conv6_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv6_1_3d)
    conv6_1_GN = GroupNormalization(groups=8, axis=-1)(conv6_1_PReLU)

    conv6_2_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv6_2_3d")(conv6_1_GN)
    conv6_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv6_2_3d)
    conv6_2_GN = GroupNormalization(groups=8, axis=-1)(conv6_2_PReLU)

    conv6_3_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv6_3_3d")(conv6_2_GN)
    conv6_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv6_3_3d)
    conv6_3_GN = GroupNormalization(groups=8, axis=-1)(conv6_3_PReLU)

    # decoder layer 3
    up7 = Conv3DTranspose(filters=32, kernel_size=5, padding='same', strides=2)(conv6_3_GN)
    add7 = concatenate([up7, conv1_1_GN], axis=-1)

    conv7_1_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv7_1_3d")(add7)
    conv7_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv7_1_3d)
    conv7_1_GN = GroupNormalization(groups=8, axis=-1)(conv7_1_PReLU)

    conv7_2_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv7_2_3d")(conv7_1_GN)
    conv7_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv7_2_3d)
    conv7_2_GN = GroupNormalization(groups=8, axis=-1)(conv7_2_PReLU)

    conv7_3_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv7_3_3d")(conv7_2_GN)
    conv7_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv7_3_3d)
    conv7_3_GN = GroupNormalization(groups=8, axis=-1)(conv7_3_PReLU)

    # last layers
    up8 = Conv3DTranspose(filters=16, kernel_size=5, padding='same', strides=2)(conv7_3_GN)

    conv8_1_3d = Conv3D(filters=8, kernel_size=5, padding='same', strides=2, name="conv8_1_3d")(up8)
    conv8_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv8_1_3d)
    conv8_1_GN = GroupNormalization(groups=8, axis=-1)(conv8_1_PReLU)

    last_layer = Conv3D(filters=1, kernel_size=5, padding='same', strides=1, name="last_layer")(conv8_1_GN)

    UNet_model = Model(inpt, last_layer)

    # UNet_model.summary(positions=[.33, .65, .75, 1.])
    
    return UNet_model
    

def define_model_regularized(cube_shape, l1_weight, l2_weight):
    """ defines a 3D CNN U-Net model with eight convolutional layers

    Args:
        cube_shape (int): cube shape in x, y, z direction

    Returns:
        UNet_model (tf.keras.Model): trainable model structure of 3D CNN U-Net
    """

    print("\n>>> DEFINE MODEL")

    inpt = Input(shape=np.zeros((cube_shape, cube_shape, cube_shape, 1)).shape)

    # encoder layer 1
    conv1_1_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv1_1_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(inpt)
    conv1_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv1_1_3d)
    conv1_1_GN = GroupNormalization(groups=8, axis=-1)(conv1_1_PReLU)

    conv1_2_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv1_2_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv1_1_GN)
    conv1_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv1_2_3d)
    conv1_2_GN = GroupNormalization(groups=8, axis=-1)(conv1_2_PReLU)

    conv1_3_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=2, name="conv1_3_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv1_2_GN)
    conv1_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv1_3_3d)
    conv1_3_GN = GroupNormalization(groups=8, axis=-1)(conv1_3_PReLU)

    # encoder layer 2
    conv2_1_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv2_1_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv1_3_GN)
    conv2_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv2_1_3d)
    conv2_1_GN = GroupNormalization(groups=8, axis=-1)(conv2_1_PReLU)

    conv2_2_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv2_2_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv2_1_GN)
    conv2_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv2_2_3d)
    conv2_2_GN = GroupNormalization(groups=8, axis=-1)(conv2_2_PReLU)

    conv2_3_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=2, name="conv2_3_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv2_2_GN)
    conv2_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv2_3_3d)
    conv2_3_GN = GroupNormalization(groups=8, axis=-1)(conv2_3_PReLU)

    # encoder layer 3
    conv3_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv3_1_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv2_3_GN)
    conv3_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv3_1_3d)
    conv3_1_GN = GroupNormalization(groups=8, axis=-1)(conv3_1_PReLU)

    conv3_2_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv3_2_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv3_1_GN)
    conv3_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv3_2_3d)
    conv3_2_GN = GroupNormalization(groups=8, axis=-1)(conv3_2_PReLU)

    conv3_3_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=2, name="conv3_3_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv3_2_GN)
    conv3_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv3_3_3d)
    conv3_3_GN = GroupNormalization(groups=8, axis=-1)(conv3_3_PReLU)

    # bottom layer
    conv4_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv4_1_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv3_3_GN)
    conv4_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv4_1_3d)
    conv4_1_GN = GroupNormalization(groups=8, axis=-1)(conv4_1_PReLU)

    # decoder layer 1
    up5 = Conv3DTranspose(filters=128, kernel_size=5, padding='same', strides=2)(conv4_1_GN)
    add5 = concatenate([up5, conv3_1_GN], axis=-1)

    conv5_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv5_1_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(add5)
    conv5_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv5_1_3d)
    conv5_1_GN = GroupNormalization(groups=8, axis=-1)(conv5_1_PReLU)

    conv5_2_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv5_2_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv5_1_GN)
    conv5_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv5_2_3d)
    conv5_2_GN = GroupNormalization(groups=8, axis=-1)(conv5_2_PReLU)

    conv5_3_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv5_3_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv5_2_GN)
    conv5_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv5_3_3d)
    conv5_3_GN = GroupNormalization(groups=8, axis=-1)(conv5_3_PReLU)

    # decoder layer 2
    up6 = Conv3DTranspose(filters=64, kernel_size=5, padding='same', strides=2)(conv5_3_GN)
    add6 = concatenate([up6, conv2_1_GN], axis=-1)

    conv6_1_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv6_1_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(add6)
    conv6_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv6_1_3d)
    conv6_1_GN = GroupNormalization(groups=8, axis=-1)(conv6_1_PReLU)

    conv6_2_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv6_2_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv6_1_GN)
    conv6_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv6_2_3d)
    conv6_2_GN = GroupNormalization(groups=8, axis=-1)(conv6_2_PReLU)

    conv6_3_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv6_3_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv6_2_GN)
    conv6_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv6_3_3d)
    conv6_3_GN = GroupNormalization(groups=8, axis=-1)(conv6_3_PReLU)

    # decoder layer 3
    up7 = Conv3DTranspose(filters=32, kernel_size=5, padding='same', strides=2)(conv6_3_GN)
    add7 = concatenate([up7, conv1_1_GN], axis=-1)

    conv7_1_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv7_1_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(add7)
    conv7_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv7_1_3d)
    conv7_1_GN = GroupNormalization(groups=8, axis=-1)(conv7_1_PReLU)

    conv7_2_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv7_2_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv7_1_GN)
    conv7_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv7_2_3d)
    conv7_2_GN = GroupNormalization(groups=8, axis=-1)(conv7_2_PReLU)

    conv7_3_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv7_3_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(conv7_2_GN)
    conv7_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv7_3_3d)
    conv7_3_GN = GroupNormalization(groups=8, axis=-1)(conv7_3_PReLU)

    # last layers
    up8 = Conv3DTranspose(filters=16, kernel_size=5, padding='same', strides=2)(conv7_3_GN)

    conv8_1_3d = Conv3D(filters=8, kernel_size=5, padding='same', strides=2, name="conv8_1_3d", kernel_regularizer=L1L2(l1=l1_weight, l2=l2_weight))(up8)
    conv8_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv8_1_3d)
    conv8_1_GN = GroupNormalization(groups=8, axis=-1)(conv8_1_PReLU)

    last_layer = Conv3D(filters=1, kernel_size=5, padding='same', strides=1, name="last_layer")(conv8_1_GN)

    UNet_model = Model(inpt, last_layer)

    # UNet_model.summary(positions=[.33, .65, .75, 1.])
    
    return UNet_model
    
def define_model_dropout(cube_shape):
    """ defines a 3D CNN U-Net model with eight convolutional layers

    Args:
        cube_shape (int): cube shape in x, y, z direction

    Returns:
        UNet_model (tf.keras.Model): trainable model structure of 3D CNN U-Net
    """

    print("\n>>> DEFINE MODEL")
    dropout_rate = 0.5
    inpt = Input(shape=np.zeros((cube_shape, cube_shape, cube_shape, 1)).shape)

    # encoder layer 1
    conv1_1_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv1_1_3d" )(inpt)
    conv1_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv1_1_3d)
    conv1_1_GN = GroupNormalization(groups=8, axis=-1)(conv1_1_PReLU)
    # add dropout
    conv1_1_dropout = Dropout(rate=dropout_rate)(conv1_1_GN)

    conv1_2_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv1_2_3d")(conv1_1_dropout)
    conv1_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv1_2_3d)
    conv1_2_GN = GroupNormalization(groups=8, axis=-1)(conv1_2_PReLU)
    # add dropout
    conv1_2_dropout = Dropout(rate=dropout_rate)(conv1_2_GN)


    conv1_3_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=2, name="conv1_3_3d")(conv1_2_dropout)
    conv1_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv1_3_3d)
    conv1_3_GN = GroupNormalization(groups=8, axis=-1)(conv1_3_PReLU)
    # add dropout
    conv1_3_dropout = Dropout(rate=dropout_rate)(conv1_3_GN)
    # encoder layer 2

    conv2_1_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv2_1_3d")(conv1_3_dropout)
    conv2_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv2_1_3d)
    conv2_1_GN = GroupNormalization(groups=8, axis=-1)(conv2_1_PReLU)
    # add dropout
    conv2_1_dropout = Dropout(rate=dropout_rate)(conv2_1_GN)

    conv2_2_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv2_2_3d")(conv2_1_dropout)
    conv2_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv2_2_3d)
    conv2_2_GN = GroupNormalization(groups=8, axis=-1)(conv2_2_PReLU)
    # add dropout
    conv2_2_dropout = Dropout(rate=dropout_rate)(conv2_2_GN)

    conv2_3_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=2, name="conv2_3_3d")(conv2_2_dropout)
    conv2_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv2_3_3d)
    conv2_3_GN = GroupNormalization(groups=8, axis=-1)(conv2_3_PReLU)
    # add dropout
    conv2_3_dropout = Dropout(rate=dropout_rate)(conv2_3_GN)

    # encoder layer 3
    conv3_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv3_1_3d")(conv2_3_dropout)
    conv3_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv3_1_3d)
    conv3_1_GN = GroupNormalization(groups=8, axis=-1)(conv3_1_PReLU)
    # add dropout
    conv3_1_dropout = Dropout(rate=dropout_rate)(conv3_1_GN)

    conv3_2_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv3_2_3d")(conv3_1_dropout)
    conv3_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv3_2_3d)
    conv3_2_GN = GroupNormalization(groups=8, axis=-1)(conv3_2_PReLU)   
    # add dropout
    conv3_2_dropout = Dropout(rate=dropout_rate)(conv3_2_GN)

    conv3_3_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=2, name="conv3_3_3d")(conv3_2_dropout)
    conv3_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv3_3_3d)
    conv3_3_GN = GroupNormalization(groups=8, axis=-1)(conv3_3_PReLU)
    # add dropout
    conv3_3_dropout = Dropout(rate=dropout_rate)(conv3_3_GN)

    # bottom layer
    conv4_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv4_1_3d")(conv3_3_dropout)
    conv4_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv4_1_3d)
    conv4_1_GN = GroupNormalization(groups=8, axis=-1)(conv4_1_PReLU)
    # add dropout
    conv4_1_dropout = Dropout(rate=dropout_rate)(conv4_1_GN)

    # decoder layer 1
    up5 = Conv3DTranspose(filters=128, kernel_size=5, padding='same', strides=2)(conv4_1_dropout)
    add5 = concatenate([up5, conv3_1_GN], axis=-1)

    conv5_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv5_1_3d")(add5)
    conv5_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv5_1_3d)
    conv5_1_GN = GroupNormalization(groups=8, axis=-1)(conv5_1_PReLU)
    # add dropout
    conv5_1_dropout = Dropout(rate=dropout_rate)(conv5_1_GN)

    conv5_2_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv5_2_3d")(conv5_1_dropout)
    conv5_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv5_2_3d)
    conv5_2_GN = GroupNormalization(groups=8, axis=-1)(conv5_2_PReLU)
    # add dropout
    conv5_2_dropout = Dropout(rate=dropout_rate)(conv5_2_GN)

    conv5_3_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv5_3_3d")(conv5_2_dropout)
    conv5_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv5_3_3d)
    conv5_3_GN = GroupNormalization(groups=8, axis=-1)(conv5_3_PReLU)
    # add dropout
    conv5_3_dropout = Dropout(rate=dropout_rate)(conv5_3_GN)

    # decoder layer 2
    up6 = Conv3DTranspose(filters=64, kernel_size=5, padding='same', strides=2)(conv5_3_dropout)
    add6 = concatenate([up6, conv2_1_GN], axis=-1)

    conv6_1_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv6_1_3d")(add6)
    conv6_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv6_1_3d)
    conv6_1_GN = GroupNormalization(groups=8, axis=-1)(conv6_1_PReLU)
    # add dropout
    conv6_1_dropout = Dropout(rate=dropout_rate)(conv6_1_GN)

    conv6_2_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv6_2_3d")(conv6_1_dropout)
    conv6_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv6_2_3d)
    conv6_2_GN = GroupNormalization(groups=8, axis=-1)(conv6_2_PReLU)
    # add dropout
    conv6_2_dropout = Dropout(rate=dropout_rate)(conv6_2_GN)

    conv6_3_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv6_3_3d")(conv6_2_dropout)
    conv6_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv6_3_3d)
    conv6_3_GN = GroupNormalization(groups=8, axis=-1)(conv6_3_PReLU)
    # add dropout
    conv6_3_dropout = Dropout(rate=dropout_rate)(conv6_3_GN)

    # decoder layer 3
    up7 = Conv3DTranspose(filters=32, kernel_size=5, padding='same', strides=2)(conv6_3_dropout)
    add7 = concatenate([up7, conv1_1_GN], axis=-1)

    conv7_1_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv7_1_3d")(add7)
    conv7_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv7_1_3d)
    conv7_1_GN = GroupNormalization(groups=8, axis=-1)(conv7_1_PReLU)
    # add dropout
    conv7_1_dropout = Dropout(rate=dropout_rate)(conv7_1_GN)

    conv7_2_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv7_2_3d")(conv7_1_dropout)
    conv7_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv7_2_3d)
    conv7_2_GN = GroupNormalization(groups=8, axis=-1)(conv7_2_PReLU)
    # add dropout
    conv7_2_dropout = Dropout(rate=dropout_rate)(conv7_2_GN)

    conv7_3_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv7_3_3d")(conv7_2_dropout)
    conv7_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv7_3_3d)
    conv7_3_GN = GroupNormalization(groups=8, axis=-1)(conv7_3_PReLU)
    # add dropout
    conv7_3_dropout = Dropout(rate=dropout_rate)(conv7_3_GN)


    # last layers
    up8 = Conv3DTranspose(filters=16, kernel_size=5, padding='same', strides=2)(conv7_3_dropout)

    conv8_1_3d = Conv3D(filters=8, kernel_size=5, padding='same', strides=2, name="conv8_1_3d")(up8)
    conv8_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv8_1_3d)
    conv8_1_GN = GroupNormalization(groups=8, axis=-1)(conv8_1_PReLU)
    # add dropout
    conv8_1_dropout = Dropout(rate=dropout_rate)(conv8_1_GN)

    last_layer = Conv3D(filters=1, kernel_size=5, padding='same', strides=1, name="last_layer")(conv8_1_dropout)

    UNet_model = Model(inpt, last_layer)

    # UNet_model.summary(positions=[.33, .65, .75, 1.])
    
    return UNet_model

def define_model_large(cube_shape):
    """ defines a 3D CNN U-Net model with ten convolutional layers

    Args:
        cube_shape (int): cube shape in x, y, z direction

    Returns:
        UNet_model (tf.keras.Model): trainable model structure of 3D CNN U-Net
    """
    
    print("\n>>> DEFINE MODEL")
    DROPOUT_RATE = 0.5
    inpt = Input(shape=np.zeros((cube_shape, cube_shape, cube_shape, 1)).shape)

    # encoder layer 1
    conv1_1_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv1_1_3d")(inpt)
    conv1_1_PReLU = ELU()(conv1_1_3d)
    conv1_1_GN = GroupNormalization(groups=8, axis=-1)(conv1_1_PReLU)
    # add dropout
    dropout1_1 = Dropout(rate=DROPOUT_RATE)(conv1_1_GN)
    
    conv1_2_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv1_2_3d")(dropout1_1)
    conv1_2_PReLU = ELU()(conv1_2_3d)
    conv1_2_GN = GroupNormalization(groups=8, axis=-1)(conv1_2_PReLU)
    # add dropout
    dropout1_2 = Dropout(rate=DROPOUT_RATE)(conv1_2_GN)
    
    conv1_3_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=2, name="conv1_3_3d")(dropout1_2)
    conv1_3_PReLU = ELU()(conv1_3_3d)
    conv1_3_GN = GroupNormalization(groups=8, axis=-1)(conv1_3_PReLU)
    # add dropout
    dropout1_3 = Dropout(rate=DROPOUT_RATE)(conv1_3_GN)
    
    # encoder layer 2
    conv2_1_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv2_1_3d")(dropout1_3)
    conv2_1_PReLU = ELU()(conv2_1_3d)
    conv2_1_GN = GroupNormalization(groups=8, axis=-1)(conv2_1_PReLU)
    # add dropout
    dropout2_1 = Dropout(rate=DROPOUT_RATE)(conv2_1_GN)
    
    conv2_2_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv2_2_3d")(dropout2_1)
    conv2_2_PReLU = ELU()(conv2_2_3d)
    conv2_2_GN = GroupNormalization(groups=8, axis=-1)(conv2_2_PReLU)
    # add dropout
    dropout2_2 = Dropout(rate=DROPOUT_RATE)(conv2_2_GN)

    conv2_3_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=2, name="conv2_3_3d")(dropout2_2)
    conv2_3_PReLU = ELU()(conv2_3_3d)
    conv2_3_GN = GroupNormalization(groups=8, axis=-1)(conv2_3_PReLU)
    # add dropout
    dropout2_3 = Dropout(rate=DROPOUT_RATE)(conv2_3_GN)
    
    # encoder layer 3
    conv3_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv3_1_3d")(dropout2_3)
    conv3_1_PReLU = ELU()(conv3_1_3d)
    conv3_1_GN = GroupNormalization(groups=8, axis=-1)(conv3_1_PReLU)
    # add dropout
    dropout3_1 = Dropout(rate=DROPOUT_RATE)(conv3_1_GN)
    
    conv3_2_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv3_2_3d")(dropout3_1)
    conv3_2_PReLU = ELU()(conv3_2_3d)
    conv3_2_GN = GroupNormalization(groups=8, axis=-1)(conv3_2_PReLU)
    # add dropout
    dropout3_2 = Dropout(rate=DROPOUT_RATE)(conv3_2_GN)

    conv3_3_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=2, name="conv3_3_3d")(dropout3_2)
    conv3_3_PReLU = ELU()(conv3_3_3d)
    conv3_3_GN = GroupNormalization(groups=8, axis=-1)(conv3_3_PReLU)
    # add dropout
    dropout3_3 = Dropout(rate=DROPOUT_RATE)(conv3_3_GN)
    
    # encoder layer 4
    conv4_1_3d = Conv3D(filters=256, kernel_size=5, padding='same', strides=1, name="conv4_1_3d")(dropout3_3)
    conv4_1_PReLU = ELU()(conv4_1_3d)
    conv4_1_GN = GroupNormalization(groups=8, axis=-1)(conv4_1_PReLU)
    # add dropout
    dropout4_1 = Dropout(rate=DROPOUT_RATE)(conv4_1_GN)
    
    conv4_2_3d = Conv3D(filters=256, kernel_size=5, padding='same', strides=1, name="conv4_2_3d")(dropout4_1)
    conv4_2_PReLU = ELU()(conv4_2_3d)
    conv4_2_GN = GroupNormalization(groups=8, axis=-1)(conv4_2_PReLU)
    # add dropout
    dropout4_2 = Dropout(rate=DROPOUT_RATE)(conv4_2_GN)
    
    conv4_3_3d = Conv3D(filters=256, kernel_size=5, padding='same', strides=2, name="conv4_3_3d")(dropout4_2)
    conv4_3_PReLU = ELU()(conv4_3_3d)
    conv4_3_GN = GroupNormalization(groups=8, axis=-1)(conv4_3_PReLU)
    # add dropout
    dropout4_3 = Dropout(rate=DROPOUT_RATE)(conv4_3_GN)
    
    # bottom layer
    conv5_1_3d = Conv3D(filters=256, kernel_size=5, padding='same', strides=1, name="conv5_1_3d")(dropout4_3)
    conv5_1_PReLU = ELU()(conv5_1_3d)
    conv5_1_GN = GroupNormalization(groups=8, axis=-1)(conv5_1_PReLU)
    
    # decoder layer 1
    up6 = Conv3DTranspose(filters=256, kernel_size=5, padding='same', strides=2)(conv5_1_GN)
    add6 = concatenate([up6, conv4_1_GN], axis=-1)

    conv6_1_3d = Conv3D(filters=256, kernel_size=5, padding='same', strides=1, name="conv6_1_3d")(add6)
    conv6_1_PReLU = ELU()(conv6_1_3d)
    conv6_1_GN = GroupNormalization(groups=8, axis=-1)(conv6_1_PReLU)
    # add dropout
    dropout6_1 = Dropout(rate=DROPOUT_RATE)(conv6_1_GN)
    
    conv6_2_3d = Conv3D(filters=256, kernel_size=5, padding='same', strides=1, name="conv6_2_3d")(dropout6_1)
    conv6_2_PReLU = ELU()(conv6_2_3d)
    conv6_2_GN = GroupNormalization(groups=8, axis=-1)(conv6_2_PReLU)
    # add dropout
    dropout6_2 = Dropout(rate=DROPOUT_RATE)(conv6_2_GN)
    
    conv6_3_3d = Conv3D(filters=256, kernel_size=5, padding='same', strides=1, name="conv6_3_3d")(dropout6_2)
    conv6_3_PReLU = ELU()(conv6_3_3d)
    conv6_3_GN = GroupNormalization(groups=8, axis=-1)(conv6_3_PReLU)
    # add dropout
    dropout6_3 = Dropout(rate=DROPOUT_RATE)(conv6_3_GN)

    # decoder layer 2
    up7 = Conv3DTranspose(filters=128, kernel_size=5, padding='same', strides=2)(dropout6_3)
    add7 = concatenate([up7, conv3_1_GN], axis=-1)

    conv7_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv7_1_3d")(add7)
    conv7_1_PReLU = ELU()(conv7_1_3d)
    conv7_1_GN = GroupNormalization(groups=8, axis=-1)(conv7_1_PReLU)
    # add dropout
    dropout7_1 = Dropout(rate=DROPOUT_RATE)(conv7_1_GN)
    
    conv7_2_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv7_2_3d")(dropout7_1)
    conv7_2_PReLU = ELU()(conv7_2_3d)
    conv7_2_GN = GroupNormalization(groups=8, axis=-1)(conv7_2_PReLU)
    # add dropout
    dropout7_2 = Dropout(rate=DROPOUT_RATE)(conv7_2_GN)
    
    conv7_3_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv7_3_3d")(dropout7_2)
    conv7_3_PReLU = ELU()(conv7_3_3d)
    conv7_3_GN = GroupNormalization(groups=8, axis=-1)(conv7_3_PReLU)
    dropout7_3 = Dropout(rate=DROPOUT_RATE)(conv7_3_GN)

    # decoder layer 3
    up8 = Conv3DTranspose(filters=64, kernel_size=5, padding='same', strides=2)(dropout7_3)
    add8 = concatenate([up8, conv2_1_GN], axis=-1)

    conv8_1_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv8_1_3d")(add8)
    conv8_1_PReLU = ELU()(conv8_1_3d)
    conv8_1_GN = GroupNormalization(groups=8, axis=-1)(conv8_1_PReLU)
    dropout8_1 = Dropout(rate=DROPOUT_RATE)(conv8_1_GN)

    conv8_2_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv8_2_3d")(dropout8_1)
    conv8_2_PReLU = ELU()(conv8_2_3d)
    conv8_2_GN = GroupNormalization(groups=8, axis=-1)(conv8_2_PReLU)
    # add dropout
    dropout8_2 = Dropout(rate=DROPOUT_RATE)(conv8_2_GN)
    
    conv8_3_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv8_3_3d")(dropout8_2)
    conv8_3_PReLU = ELU()(conv8_3_3d)
    conv8_3_GN = GroupNormalization(groups=8, axis=-1)(conv8_3_PReLU)
    # add dropout
    dropout8_3 = Dropout(rate=DROPOUT_RATE)(conv8_3_GN)
    
    # decoder layer 4
    up9 = Conv3DTranspose(filters=32, kernel_size=5, padding='same', strides=2)(dropout8_3)
    add9 = concatenate([up9, conv1_1_GN], axis=-1)

    conv9_1_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv9_1_3d")(add9)
    conv9_1_PReLU = ELU()(conv9_1_3d)
    conv9_1_GN = GroupNormalization(groups=8, axis=-1)(conv9_1_PReLU)
    # add dropout
    dropout9_1 = Dropout(rate=DROPOUT_RATE)(conv9_1_GN)
    
    conv9_2_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv9_2_3d")(dropout9_1)
    conv9_2_PReLU = ELU()(conv9_2_3d)
    conv9_2_GN = GroupNormalization(groups=8, axis=-1)(conv9_2_PReLU)
    # add dropout
    dropout9_2 = Dropout(rate=DROPOUT_RATE)(conv9_2_GN)
    
    conv9_3_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv9_3_3d")(dropout9_2)
    conv9_3_PReLU = ELU()(conv9_3_3d)
    conv9_3_GN = GroupNormalization(groups=8, axis=-1)(conv9_3_PReLU)

    # last layers
    up10 = Conv3DTranspose(filters=16, kernel_size=5, padding='same', strides=2)(conv9_3_GN)

    conv10_1_3d = Conv3D(filters=8, kernel_size=5, padding='same', strides=2, name="conv10_1_3d")(up10)
    conv10_1_PReLU = ELU()(conv10_1_3d)
    conv10_1_GN = GroupNormalization(groups=8, axis=-1)(conv10_1_PReLU)

    last_layer = Conv3D(filters=1, kernel_size=5, padding='same', strides=1, name="last_layer")(conv10_1_GN)

    UNet_model = Model(inpt, last_layer)

    UNet_model.summary(positions=[.33, .65, .75, 1.])
    
    return UNet_model


def define_model_two_channel(cube_shape):
    """ defines a 3D CNN U-Net model with eight convolutional layers

    Args:
        cube_shape (int): cube shape in x, y, z, direction

    Returns:
        UNet_model (tf.keras.Model): trainable model structure of 3D CNN U-Net
    
        Note the model should predict two channel for output one for amplitude and one for phase
    """

    print("\n>>> DEFINE MODEL")
    DROPOUT_RATE = 0.5
    inpt = Input(shape=np.zeros((cube_shape, cube_shape, cube_shape, 1)).shape)

    # encoder layer 1
    conv1_1_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv1_1_3d")(inpt)
    conv1_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv1_1_3d)
    conv1_1_GN = GroupNormalization(groups=8, axis=-1)(conv1_1_PReLU)
    dropout1_1 = Dropout(rate=DROPOUT_RATE)(conv1_1_GN)

    conv1_2_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv1_2_3d")(dropout1_1)
    conv1_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv1_2_3d)
    conv1_2_GN = GroupNormalization(groups=8, axis=-1)(conv1_2_PReLU)
    dropout1_2 = Dropout(rate=DROPOUT_RATE)(conv1_2_GN)

    conv1_3_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=2, name="conv1_3_3d")(dropout1_2)
    conv1_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv1_3_3d)
    conv1_3_GN = GroupNormalization(groups=8, axis=-1)(conv1_3_PReLU)
    dropout1_3 = Dropout(rate=DROPOUT_RATE)(conv1_3_GN)
    
    # encoder layer 2
    conv2_1_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv2_1_3d")(dropout1_3)
    conv2_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv2_1_3d)
    conv2_1_GN = GroupNormalization(groups=8, axis=-1)(conv2_1_PReLU)
    dropout2_1 = Dropout(rate=DROPOUT_RATE)(conv2_1_GN)
    
    conv2_2_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv2_2_3d")(dropout2_1)
    conv2_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv2_2_3d)
    conv2_2_GN = GroupNormalization(groups=8, axis=-1)(conv2_2_PReLU)
    dropout2_2 = Dropout(rate=DROPOUT_RATE)(conv2_2_GN)

    conv2_3_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=2, name="conv2_3_3d")(dropout2_2)
    conv2_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv2_3_3d)
    conv2_3_GN = GroupNormalization(groups=8, axis=-1)(conv2_3_PReLU)
    dropout2_3 = Dropout(rate=DROPOUT_RATE)(conv2_3_GN)
    
    # encoder layer 3
    conv3_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv3_1_3d")(dropout2_3)
    conv3_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv3_1_3d)
    conv3_1_GN = GroupNormalization(groups=8, axis=-1)(conv3_1_PReLU)
    dropout3_1 = Dropout(rate=DROPOUT_RATE)(conv3_1_GN)
    
    conv3_2_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv3_2_3d")(dropout3_1)
    conv3_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv3_2_3d)
    conv3_2_GN = GroupNormalization(groups=8, axis=-1)(conv3_2_PReLU)
    dropout3_2 = Dropout(rate=DROPOUT_RATE)(conv3_2_GN)
    
    conv3_3_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=2, name="conv3_3_3d")(dropout3_2)
    conv3_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv3_3_3d)
    conv3_3_GN = GroupNormalization(groups=8, axis=-1)(conv3_3_PReLU)
    dropout3_3 = Dropout(rate=DROPOUT_RATE)(conv3_3_GN)

    # bottom layer
    conv4_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv4_1_3d")(dropout3_3)
    conv4_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv4_1_3d)
    conv4_1_GN = GroupNormalization(groups=8, axis=-1)(conv4_1_PReLU)

    # decoder layer 1
    up5 = Conv3DTranspose(filters=128, kernel_size=5, padding='same', strides=2)(conv4_1_GN)
    add5 = concatenate([up5, conv3_1_GN], axis=-1)

    conv5_1_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv5_1_3d")(add5)
    conv5_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv5_1_3d)
    conv5_1_GN = GroupNormalization(groups=8, axis=-1)(conv5_1_PReLU)
    dropout5_1 = Dropout(rate=DROPOUT_RATE)(conv5_1_GN)

    conv5_2_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv5_2_3d")(dropout5_1)
    conv5_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv5_2_3d)
    conv5_2_GN = GroupNormalization(groups=8, axis=-1)(conv5_2_PReLU)
    dropout5_2 = Dropout(rate=DROPOUT_RATE)(conv5_2_GN)

    conv5_3_3d = Conv3D(filters=128, kernel_size=5, padding='same', strides=1, name="conv5_3_3d")(dropout5_2)
    conv5_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv5_3_3d)
    conv5_3_GN = GroupNormalization(groups=8, axis=-1)(conv5_3_PReLU)
    dropout5_3 = Dropout(rate=DROPOUT_RATE)(conv5_3_GN)

    # decoder layer 2
    up6 = Conv3DTranspose(filters=64, kernel_size=5, padding='same', strides=2)(dropout5_3)
    add6 = concatenate([up6, conv2_1_GN], axis=-1)

    conv6_1_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv6_1_3d")(add6)
    conv6_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv6_1_3d)
    conv6_1_GN = GroupNormalization(groups=8, axis=-1)(conv6_1_PReLU)
    dropout6_1 = Dropout(rate=DROPOUT_RATE)(conv6_1_GN)

    conv6_2_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv6_2_3d")(dropout6_1)
    conv6_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv6_2_3d)
    conv6_2_GN = GroupNormalization(groups=8, axis=-1)(conv6_2_PReLU)
    dropout6_2 = Dropout(rate=DROPOUT_RATE)(conv6_2_GN)
    
    conv6_3_3d = Conv3D(filters=64, kernel_size=5, padding='same', strides=1, name="conv6_3_3d")(dropout6_2)
    conv6_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv6_3_3d)
    conv6_3_GN = GroupNormalization(groups=8, axis=-1)(conv6_3_PReLU)
    dropout6_3 = Dropout(rate=DROPOUT_RATE)(conv6_3_GN)
    
    # decoder layer 3
    up7 = Conv3DTranspose(filters=32, kernel_size=5, padding='same', strides=2)(dropout6_3)
    add7 = concatenate([up7, conv1_1_GN], axis=-1)

    conv7_1_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv7_1_3d")(add7)
    conv7_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv7_1_3d)
    conv7_1_GN = GroupNormalization(groups=8, axis=-1)(conv7_1_PReLU)
    dropout7_1 = Dropout(rate=DROPOUT_RATE)(conv7_1_GN)

    conv7_2_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv7_2_3d")(dropout7_1)
    conv7_2_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv7_2_3d)
    conv7_2_GN = GroupNormalization(groups=8, axis=-1)(conv7_2_PReLU)
    dropout7_2 = Dropout(rate=DROPOUT_RATE)(conv7_2_GN)
    
    conv7_3_3d = Conv3D(filters=32, kernel_size=5, padding='same', strides=1, name="conv7_3_3d")(dropout7_2)
    conv7_3_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv7_3_3d)
    conv7_3_GN = GroupNormalization(groups=8, axis=-1)(conv7_3_PReLU)
    dropout7_3 = Dropout(rate=DROPOUT_RATE)(conv7_3_GN)
    
    # last layers
    up8 = Conv3DTranspose(filters=16, kernel_size=5, padding='same', strides=1)(dropout7_3)

    conv8_1_3d = Conv3D(filters=8, kernel_size=5, padding='same', strides=1, name="conv8_1_3d")(up8)
    conv8_1_PReLU = PReLU(alpha_initializer='zeros', shared_axes=[1,2,3])(conv8_1_3d)
    conv8_1_GN = GroupNormalization(groups=8, axis=-1)(conv8_1_PReLU)

    # Last layer should have two channels
    last_layer = Conv3D(filters=2, kernel_size=5, padding='same', strides=1, name="last_layer")(conv8_1_GN)

    UNet_model = Model(inpt, last_layer)

    # UNet_model.summary(positions=[.33, .65, .75, 1.])
    
    return UNet_model
    