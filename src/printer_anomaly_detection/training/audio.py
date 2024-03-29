import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess

import tensorflow as tf
import yaml
from printer_anomaly_detection.model.cae import CAE
from printer_anomaly_detection.dataset.audio import load_audio_dataset_split, Split, get_normalization_stats
from datetime import datetime

import warnings

@dataclass
class AudioTrainingConfig:
    phase: str
    epochs: int
    loss: str
    latent_dim: int
    name: str
    timestring: str
    renorm: bool
    activation: str
    last_activation: str
    batch_size: int
    data_steps: int
    dropout: float
    learning_rate: float
    commit_hash: str
    decay_factor: float
    shuffle_data: bool
    disable_normalization: bool
    scale: bool

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    def get_git_revision_short_hash() -> str:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

    parser = argparse.ArgumentParser(
        description="Preprocessing Script"
    )
    parser.add_argument(
        "--name",
        type=str,
        metavar='NAME',
        required=True,
        help="Name of the training instance"
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
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--loss",
        type=str,
        default='mse',
    )
    parser.add_argument(
        "--renorm",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--scale",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--disable_normalization",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--last_activation",
        type=str,
        default='tanh'
    )
    parser.add_argument(
        "--activation",
        type=str,
        default='relu',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--data_steps",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--decay_factor",
        type=float,
        default=2.,
    )
    parser.add_argument(
        "--shuffle_data",
        action='store_true',
        default=False
    )
    timestring = datetime.now().isoformat(timespec='seconds')

    args = parser.parse_args()

    commit_hash = get_git_revision_short_hash()
    
    name = f'{args.name}-{commit_hash}'
    phase = args.phase
    latent_dim = args.latent_dim
    epochs = args.epochs
    loss = args.loss
    renorm = args.renorm
    activation = args.activation
    last_activation = args.last_activation
    batch_size = args.batch_size
    data_steps = args.data_steps
    dropout = args.dropout
    learning_rate = args.learning_rate
    decay_factor = args.decay_factor
    shuffle_data = args.shuffle_data
    disable_normalization = args.disable_normalization
    scale = args.scale

    assert not scale or disable_normalization, "Cannot scale and normalize at the same time"

    config = AudioTrainingConfig(phase=phase, epochs=epochs, loss=loss, latent_dim=latent_dim, name=name, timestring=timestring, renorm=renorm, last_activation=last_activation, activation=activation, batch_size=batch_size, data_steps=data_steps, dropout=dropout, learning_rate=learning_rate, commit_hash=commit_hash, decay_factor=decay_factor, shuffle_data=shuffle_data, disable_normalization=disable_normalization, scale=scale)

    def l1norm(y_true,y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def l2norm(y_true,y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    loss_mappings = {
        'kld': tf.keras.losses.KLDivergence(),
        'bce': tf.keras.losses.BinaryCrossentropy(),
        'l2norm': l2norm,
        'l1norm': l1norm
    }
    
    loss = loss_mappings.get(loss, loss)

    dataset_path = Path(args.dataset_folder)

    train_dataset, test_dataset = load_audio_dataset_split(dataset_path, phase, Split.TRAIN, window_size=256, step_size=data_steps, shuffle_data=shuffle_data, scale=scale), \
                                  load_audio_dataset_split(dataset_path, phase, Split.TEST, window_size=256, step_size=data_steps, scale=scale)

    #assert len(list(train_dataset.take(1))) > 0, "No training data found"

    #train_dataset = tf.data.Dataset.zip((train_dataset, train_dataset))
    #test_dataset = tf.data.Dataset.zip((test_dataset, test_dataset))

    mean, var = get_normalization_stats(dataset_path, phase) if not disable_normalization else (None, None)

    model = CAE(latent_dim=latent_dim, renorm=renorm, activation=activation, dropout=dropout, mean=mean, var=var, last_activation=last_activation)
    model.summary()

    dataset_len = 3164.8 #34967.6
    decay_base = (532000./3164.8) * dataset_len

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_factor*decay_base/(data_steps*batch_size),
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=[loss], metrics=['mae', 'mse', 'crossentropy', l1norm, l2norm])
    #model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=[loss], metrics=['mae', 'crossentropy'])

    train_dataset = train_dataset.map(lambda x: (x, x)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(lambda x: (x, x)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    assert len(list(test_dataset.take(1))) > 0

    # write config to yaml:
    Path('trainings/').mkdir(exist_ok=True)
    with open(f'trainings/{timestring}-{name}-config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # todo: learning rate 
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, validation_freq=1, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=f'logs/{timestring}-{name}'),
        # after each epoch
        tf.keras.callbacks.ModelCheckpoint(filepath=f'checkpoints/{timestring}-{name}/{name}' + '.{epoch:02d}-{val_loss:.4f}.h5', save_freq='epoch', verbose=1, monitor='val_loss', save_weights_only=True)
    ])
