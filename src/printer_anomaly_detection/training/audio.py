import argparse
from pathlib import Path

import tensorflow as tf
from printer_anomaly_detection.model.cae import CAE
from printer_anomaly_detection.dataset.audio import load_audio_dataset_split, Split

import warnings

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    parser = argparse.ArgumentParser(
        description="Preprocessing Script"
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        metavar='DIRECTORY',
        required=True,
        help="Path to the data folder containing all data"
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        help="Phase of the data (original, exhaust, direct_drive)"
    )

    args = parser.parse_args()
    
    phase = args.phase
    dataset_path = Path(args.dataset_folder)

    train_dataset, test_dataset = load_audio_dataset_split(dataset_path, phase, Split.TRAIN, window_size=256, step_size=32), \
                                  load_audio_dataset_split(dataset_path, phase, Split.TEST, window_size=256, step_size=32)

    #assert len(list(train_dataset.take(1))) > 0, "No training data found"

    #train_dataset = tf.data.Dataset.zip((train_dataset, train_dataset))
    #test_dataset = tf.data.Dataset.zip((test_dataset, test_dataset))

    model = CAE()
    model.summary()

    def image_loss(y_true,y_pred):
        return tf.norm(y_true - y_pred)

    model.compile(optimizer='adam', loss=['mae'], metrics=['mae', 'crossentropy'])

    train_dataset = train_dataset.map(lambda x: (x, x)).batch(64).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(lambda x: (x, x)).batch(64).prefetch(tf.data.AUTOTUNE)

    assert len(list(test_dataset.take(1))) > 0

    # todo: learning rate 
    model.fit(train_dataset, epochs=200, validation_data=test_dataset, validation_freq=1, callbacks=[
        tf.keras.callbacks.TensorBoard(),
        # after each epoch
        tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/', save_weights_only=True, save_freq='epoch')
    ])
