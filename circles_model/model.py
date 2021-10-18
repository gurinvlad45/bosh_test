import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from circles_model.circle import Circle

import cv2

IMAGE_ORDERING = 'channels_last'


def generate_and_save_images(model, epoch, step, test_input):
    """Helper function to plot our 16 images

    Args:

    model -- the decoder model
    epoch -- current epoch number during training
    step -- current step number during training
    test_input -- random tensor with shape (16, LATENT_DIM)
    """
    predictions = model.predict(test_input)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        img = predictions[i, :, :, :] * 255
        img = img.astype('int32')
        plt.imshow(img)
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    fig.suptitle("epoch: {}, step: {}".format(epoch, step))
    plt.savefig('image_at_epoch_{:04d}_step{:04d}.png'.format(epoch, step))
    plt.show()


def conv_block(input, filters, kernel_size, latent_dim):
    """
    Args:
      input (tensor) -- batch of images or features
      filters (int) -- number of filters of the Conv2D layers
      kernel_size (int) -- kernel_size setting of the Conv2D layers

    Returns:
      (tensor) max pooled and batch-normalized features of the input
    :param input:
    :param filters:
    :param kernel_size:
    :return:
    """

    # use the functional syntax to stack the layers as shown in the diagram above
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', data_format=IMAGE_ORDERING)(input)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_dim, activation='relu', name='latent_variable')(x)
    return x


def circle_encoder(input_height=64, input_width=64, latent_dim=1):
    image_input = tf.keras.layers.Input(shape=(input_height, input_width, 1))
    kernel_size = 64
    x = conv_block(image_input, filters=1, kernel_size=kernel_size, latent_dim=latent_dim)
    return x, image_input


def circle_decoder(decoder_input):
    x = tf.keras.layers.Reshape((1, 1, 1), name='decode_reshape')(decoder_input)
    x = tf.keras.layers.Conv2DTranspose(1, kernel_size=(64, 64),  strides=(1, 1), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    return x


# TODO: architecture optimization
def dense_autoencoder(input_height=64, input_width=64, latent_dim=1):
    input_image = tf.keras.layers.Input(shape=(input_height*input_width, ))
    encoded = tf.keras.layers.Dense(256, activation='relu')(input_image)
    encoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(8, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(latent_dim, activation='relu', name='bottleneck')(encoded)
    decoded = tf.keras.layers.Dense(8, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(32, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(64, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(256, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(64 * 64, activation='sigmoid')(decoded)
    autoencoder = tf.keras.Model(input_image, decoded)
    return autoencoder


dense_autoencoder_model = dense_autoencoder()
encoder_output, img_input = circle_encoder()
decoder_output = circle_decoder(encoder_output)

model = tf.keras.Model(inputs=img_input, outputs=decoder_output)
model.summary()
dense_autoencoder_model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 50

# create training_dataset
train = np.stack([Circle(diameter=2*i).create_circle_image() for i in range(1, 32)] * 10000)
dense_autoencoder_model.compile(optimizer=optimizer, loss='binary_crossentropy')
dense_autoencoder_model.fit(train, train, epochs=epochs, batch_size=256, shuffle=True)
# model.compile(optimizer=optimizer, loss='binary_crossentropy')
# model.fit(train, train, epochs=epochs, batch_size=256, shuffle=True)
predictions = dense_autoencoder_model.predict(train)[:10]


for i, image in enumerate(train[:10]):
    cv2.imwrite(f"train{i}.png", image.reshape((64, 64)) * 255.0)


for i, image in enumerate(predictions):
    cv2.imwrite(f"predictions{i}.png", image.reshape((64, 64)) * 255.0)
