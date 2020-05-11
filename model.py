import tensorflow

# tensorflow 2
if tensorflow.__version__ > '2.':
    # Enable dynamic memory allocation on GPU in tf2-------------------------------------------
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tensorflow.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tensorflow.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    # -----------------------------------------------------------------------------------------
    from tensorflow.keras.models import *
    from tensorflow.keras.layers import *
    from tensorflow.keras.optimizers import *
    from tensorflow.keras.losses import *
    from tensorflow.keras import backend as K
else:
    # tensoflow 1 + keras

    # Enable dynamic memory allocation on GPU in tf1-------------------------------------------
    from keras.backend.tensorflow_backend import set_session
    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    #config.log_device_placement = True  # to log device placement (on which device the operation ran) - for debugging
    sess = tensorflow.Session(config=config)
    set_session(sess)
    # -----------------------------------------------------------------------------------------
    from keras.models import *
    from keras.layers import *
    from keras.optimizers import *
    from keras.losses import *
    from keras import backend as K

smooth = 1.

#special metrics for FCN training on small blobs - Dice
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth, thresh):
    y_pred = y_pred > thresh
    y_pred = K.cast(y_pred, K.floatx())
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef_softmax(y_true, y_pred, smooth):
    # y_pred = y_pred == channel_id
    # y_pred = K.cast(y_pred, K.floatx())
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_loss(smooth, thresh, channel_id):
  def iou_dice(y_true, y_pred):
    return 1.-iou_coef(y_true[:,:,:,channel_id], y_pred[:,:,:,channel_id], smooth, thresh)
  return iou_dice

def iou_metric(smooth, thresh, channel_id):
  def iou_channel(y_true, y_pred):
    return iou_coef(y_true[:,:,:,channel_id], y_pred[:,:,:,channel_id], smooth, thresh)
  iou_channel.__name__ = iou_channel.__name__ + "_" + str(channel_id)
  return iou_channel

def iou_metric_softmax(smooth, channel_id):
  def iou(y_true, y_pred):

    y_pred_softmax = K.argmax(y_pred)
    y_pred_softmax = y_pred_softmax == channel_id
    y_pred_softmax = K.cast(y_pred_softmax, K.floatx())

    y_true_softmax = K.argmax(y_true)
    y_true_softmax = y_true_softmax == channel_id
    y_true_softmax = K.cast(y_true_softmax, K.floatx())

    return iou_coef_softmax(y_true_softmax, y_pred_softmax, smooth)
  iou.__name__ = iou.__name__ + "_" + str(channel_id)
  return iou

def dice_metric(smooth, thresh, channel_id):
  def dice_channel(y_true, y_pred):
    return dice_coef(y_true[:,:,:,channel_id], y_pred[:,:,:,channel_id], smooth, thresh)
  dice_channel.__name__ = dice_channel.__name__ + "_" + str(channel_id)
  return dice_channel

#loss metrics for FCN training on base of Dice
def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def dice_coef_multilabel(num_labels):
    def dice_coef_multi(y_true, y_pred):
        dice=0
        for index in range(num_labels):
            if index == 0:
                continue
            dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index]) # output tensor have shape (batch_size,
        return dice/(num_labels-1)                                           # width, height, numLabels)
    return dice_coef_multi

def dice_loss_multilabel(num_labels):
    def dice_loss_multi(y_true, y_pred):
        dice_coef_f = dice_coef_multilabel(num_labels)
        return 1. - dice_coef_f(y_true, y_pred)

    return dice_loss_multi

def dice_sparsed(smooth = 1e-5, sparse_thresh = 10):
    # useful for datasets with sparse object classes (only few images contains object class)
    # sparse_thresh - threshold for selection images with non zero number of pixels with object class
    # if number of pixels with object class (n_true_pixels) < sparse_thresh than we make dice equal to 0,
    # otherwise to common dice
    def dice_coef_sparsed(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        n_true_pixels = K.sum(y_true_f * y_true_f)
        dice_common = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        dice_coef_sparsed_value = K.switch(n_true_pixels < sparse_thresh, 0.0, dice_common)
        return dice_coef_sparsed_value
    return dice_coef_sparsed

def dice_sparsed_metric(smooth, thresh, channel_id):
  def dice_s(y_true, y_pred):
    dice_sparsed_f = dice_sparsed(smooth, thresh)
    return dice_sparsed_f(y_true[:,:,:,channel_id], y_pred[:,:,:,channel_id])
  dice_s.__name__ = dice_s.__name__ + "_" + str(channel_id)
  return dice_s

def sparsed_dice_loss_multilabel(num_labels, smooth = 1e-5, sparse_thresh=10, coefs=None):
    def sparsed_dice_loss_multi(y_true, y_pred):

        sparsed_dice_sum = 0
        coef_condition = (coefs is not None) and (len(coefs) == num_labels)
        for index in range(num_labels):
            # if index == 0:
            #     continue
            dice_sparsed_f = dice_sparsed(smooth, sparse_thresh)
            coef = 1.0
            if coef_condition:
                coef = coefs[index]
            sparsed_dice_sum += coef*dice_sparsed_f(y_true[:, :, :, index], y_pred[:, :, :, index])

        return 1. - sparsed_dice_sum / (num_labels)
    return sparsed_dice_loss_multi


def categorical_crossentropy_loss(y_true, y_pred):
    cce = CategoricalCrossentropy()
    ccel = cce(y_true, y_pred)
    return ccel

def background_dice_loss(y_true, y_pred):
    bdl = dice_coef_loss(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    return bdl

def loss_function(y_true, y_pred):
    sum_loss = categorical_crossentropy_loss(y_true, y_pred) + background_dice_loss(y_true, y_pred)
    return sum_loss

def loss_function_multilabel(num_labels, smooth=1e-5, sparse_thresh=10, coefs=[1, 1.44, 1.49, 4.4, 44.9, 62.2]):
    def loss_multilabel(y_true, y_pred):
        sparsed_dice_loss_multilabel_f = sparsed_dice_loss_multilabel(num_labels=num_labels,
                                                                      smooth=smooth,
                                                                      sparse_thresh=sparse_thresh,
                                                                      coefs=coefs)
        sum_loss =  categorical_crossentropy_loss(y_true, y_pred) + sparsed_dice_loss_multilabel_f(y_true, y_pred)
        return sum_loss
    return loss_multilabel

def dice_0(y_true, y_pred):
    return dice_coef(y_true[:,:,:,0], y_pred[:,:,:,0])

def unet_light_mct(input_size = (256,256,1),n_classes = 1):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv11 = Conv2D(32, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv11)
    conv2 = Conv2D(32, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool1)
    conv21 = Conv2D(32, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv2)
    conv22 = Conv2D(32, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv21)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)
    conv3 = Conv2D(128, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)

    # up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop3))
    # Jason Brownlee. How to use the UpSampling2D and Conv2DTranspose Layers in Keras. 2019 https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/
    # TensorRT Support Matrix. TensorRT 5.1.5. https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt-515/tensorrt-support-matrix/index.html
    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(drop3)
    merge8 = concatenate([conv22, up8], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(conv8)
    merge9 = concatenate([conv11, up9], axis=3)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(n_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10, name='unet_light_mct')

    return model

def unet(input_size = (256,256,1), n_classes = 6):

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    #conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(n_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10, name='unet')

    return model
