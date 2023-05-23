'''
This script extracts VGG features from BCSS dataset
'''

network_name = 'vgg16' #[ 'incv3', 'resnet50', 'vgg16', 'vgg19' ]
import time
#import myutils
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, AveragePooling2D, GlobalAveragePooling2D
from keras import backend as K

from keras.models import Model
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.resnet50     import ResNet50
from keras.applications.vgg16        import VGG16, preprocess_input
#from keras.applications.vgg19        import VGG19

# input_shape = {
#     'incv3'   : (299,299,3),
#     'resnet50': (224,224,3),
#     'vgg16'   : (224,224,3),
#     'vgg19'   : (224,224,3)
# }[selected_network]

input_shape = (224,224,3)

#from keras.datasets import cifar10
from bcss_patched_config import dataset, outputs_dir, project_name
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, y_train = dataset.train_dataset
X_test, y_test = dataset.test_dataset
X, y = dataset.full_dataset

n_training = X_train.shape[0]
n_testing = X_test.shape[0]

y_train = y_train.flatten()
y_test  = y_test.flatten()

print( X_train.shape, y_train.shape,  X_test.shape, y_test.shape )

#exit()
#selected_network = 'vgg16'

def create_model_vgg16():
    tf_input = Input(shape=input_shape)
    model = VGG16(input_tensor=tf_input, include_top=False)
    #output_pooled = tf.keras.layers.UpSampling2D(size=(32, 32))
    output_pooled = AveragePooling2D((7, 7))(model.output)
    return Model(model.input, output_pooled)

# tensorflow placeholder for batch of images from CIFAR10 dataset
batch_of_images_placeholder = tf.placeholder("uint8", (None, 7, 7, 3)) #tf.placeholder("uint8", (None, 32, 32, 3))

print("input_shape[:2]", input_shape[:2])
batch_size = 16
tf_resize_op = tf.image.resize_images(batch_of_images_placeholder, (input_shape[:2]), method=0)

def data_generator(sess,data,labels):
    def generator():
        start = 0
        end = start + batch_size
        n = data.shape[0]
        while True:
            batch_of_images_resized = sess.run(tf_resize_op, {batch_of_images_placeholder: data[start:end]})
            batch_of_images__preprocessed = preprocess_input(batch_of_images_resized)
            batch_of_labels = labels[start:end]
            start += batch_size
            end   += batch_size
            if start >= n:
                start = 0
                end = batch_size
            yield (batch_of_images__preprocessed, batch_of_labels)
    return generator


with tf.Session() as sess:
    # setting tensorflow session to Keras
    K.set_session(sess)
    # setting phase to training
    K.set_learning_phase(0)  # 0 - test,  1 - train

    model = create_model_vgg16()

    data_train_gen = data_generator(sess, X_train, y_train)
    ftrs_training = model.predict_generator(data_train_gen(), n_training/batch_size, verbose=1)

    data_test_gen = data_generator(sess, X_test, y_test)
    ftrs_testing = model.predict_generator(data_test_gen(), n_testing/batch_size, verbose=1)
    

features_training = np.array( [ftrs_training[i].flatten() for i in range(n_training)] )
features_testing  = np.array( [ftrs_testing[i].flatten()  for i in range(n_testing )] )

np.savez_compressed("/work/saloua/uncertainty-main/outputs/bcss_patched/bcc_patched_{}-keras_features.npz".format(selected_network), \
                    features_training=features_training, \
                    features_testing=features_testing,   \
                    labels_training=y_train,             \
                    labels_testing=y_test)

