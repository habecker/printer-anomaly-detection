from typing import Tuple

import tensorflow as tf

def CAE(latent_dim: int = 256, input_shape: Tuple[int, int, int] = (256, 256, 1), activation='relu', renorm=False, dropout = 0.0, mean: float = None, var: float = None, last_activation=None) -> tf.keras.Model:
    normalization = None

    if mean and var:
        normalization = tf.keras.layers.Normalization(axis=None, mean=mean, variance=var)

    layers = [
             tf.keras.layers.InputLayer(input_shape=input_shape),
    ]
    
    if normalization:
        layers += [normalization]

    layers += [
             tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation=activation),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation=activation),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation=activation),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation=activation),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation=activation),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(latent_dim),
             tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
             tf.keras.layers.Dense(units=4096, activation=activation),
             tf.keras.layers.Reshape(target_shape=(8, 8, 64)),
             tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation=activation),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation=activation),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation=activation),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation=activation),
             tf.keras.layers.BatchNormalization(renorm=renorm),
             tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', activation=activation),
         ]

    if normalization:
        layers.append(tf.keras.layers.Normalization(axis=None, mean=mean, variance=var, invert=True))

    return tf.keras.Sequential(layers)

if __name__ == '__main__':
    model = CAE()
    model.build(input_shape=(None, 256, 256, 1))
    model.summary()
