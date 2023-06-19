from typing import Tuple

import tensorflow as tf

def CAE(latent_dim: int = 200, input_shape: Tuple[int, int, int] = (256, 256, 1)) -> tf.keras.Model:
    return tf.keras.Sequential(
         [
             tf.keras.layers.InputLayer(input_shape=input_shape),
             tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'),
             tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(latent_dim),
             tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
             tf.keras.layers.Dense(units=4096, activation='relu'),
             tf.keras.layers.Reshape(target_shape=(8, 8, 64)),
             tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', activation='sigmoid'),
         ]
     )


if __name__ == '__main__':
    model = CAE()
    model.build(input_shape=(None, 256, 256, 1))
    model.summary()
