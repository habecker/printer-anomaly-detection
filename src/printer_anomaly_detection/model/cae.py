from typing import Tuple

import tensorflow as tf

def CAE(latent_dim: int = 256, input_shape: Tuple[int, int, int] = (256, 256, 1), last_activation='sigmoid', renorm=False, dropout = 0.0) -> tf.keras.Model:
    #normalization = tf.keras.layers.Normalization(axis=(0, 1))
    return tf.keras.Sequential(
         [
             tf.keras.layers.InputLayer(input_shape=input_shape),
    #         normalization,
             tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(latent_dim),
             tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
             tf.keras.layers.Dense(units=4096, activation='relu'),
             tf.keras.layers.Reshape(target_shape=(8, 8, 64)),
             tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', activation=last_activation),
      #       tf.keras.layers.Normalization(axis=(0, 1), mean=normalization.mean, variance=normalization.variance, invert=True),
         ]
     )


if __name__ == '__main__':
    model = CAE()
    model.build(input_shape=(None, 256, 256, 1))
    model.summary()
