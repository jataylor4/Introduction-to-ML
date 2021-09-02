import tensorflow as tf
import pylab as plt
import numpy as np

def generator(images):
    while True:
        for im in images:
            noised = im + np.random.normal(0., 0.1, im.shape)
            noised = noised[np.newaxis]
            yield noised, im[np.newaxis]

def build_unet_model():

    # Encoder
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(inputs)
    l1 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)

    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", strides=2)(l1)
    l2 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", strides=2)(l2)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)

    # Decoder part
    x = tf.keras.layers.Conv2DTranspose(16, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = tf.keras.layers.Concatenate()([x, l2])
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)

    x = tf.keras.layers.Conv2DTranspose(8, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = tf.keras.layers.Concatenate()([x, l1])
    x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)

    # Output Layer
    x = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(inputs, x)
    model.compile(loss="binary_crossentropy", optimizer="adam")

    model.summary()

    return model

def build_unet_with_skip_connections_model():

    # Encoder
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(inputs)
    l1 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", strides=2)(l1)
    l2 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", strides=2)(l2)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)

    # Decoder part
    x = tf.keras.layers.Conv2DTranspose(16, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = tf.keras.layers.Add()([x, l2])
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(8, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = tf.keras.layers.Add()([x, l1])
    x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)

    # Output Layer
    x = tf.keras.layers.Conv2D(1, (3,3), activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(inputs, x)
    model.compile(loss="binary_crossentropy", optimizer="adam")

    model.summary()

    return model

if __name__ == "__main__":

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    print(train_images.shape, train_labels.shape)

    # image preprocessing
    NB_IMAGES_TO_USE = 1000

    train_images = train_images[:NB_IMAGES_TO_USE] / 255.0
    test_images = test_images[:NB_IMAGES_TO_USE] / 255.0

    print(train_images.shape, test_images.shape)

    plt.subplot(121)
    plt.imshow(train_images[0], cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(train_images[0] + np.random.normal(0., 0.1, (28, 28)), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())
    plt.title("Noised")
    plt.savefig("figures/original vs noised - MNIST")

    unet_model = build_unet_model()
    unet_skip_model = build_unet_with_skip_connections_model()

    # Training
    cbk = tf.keras.callbacks.TensorBoard('mnist_unet_2')
    train_gen = generator(train_images[:,:,:,np.newaxis])
    unet_model.fit(train_gen, epochs=10, steps_per_epoch=20, callbacks=[cbk])

    # Testing
    test_im = test_images[5][np.newaxis, :,:, np.newaxis] + np.random.normal(0., 0.1, (1, 28, 28, 1))
    outputs = unet_model.predict(test_im)

    # Training
    cbk2 = tf.keras.callbacks.TensorBoard('mnist_unet_2')
    train_gen = generator(train_images[:,:,:,np.newaxis])
    unet_skip_model.fit(train_gen, epochs=10, steps_per_epoch=20, callbacks=[cbk2])

    # Testing
    test_im = test_images[5][np.newaxis, :,:, np.newaxis] + np.random.normal(0., 0.1, (1, 28, 28, 1))
    outputs = unet_skip_model.predict(test_im)



