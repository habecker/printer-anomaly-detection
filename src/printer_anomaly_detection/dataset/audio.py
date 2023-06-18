from functools import cache
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Set

import dataclass_csv
import librosa
import numpy as np

from printer_anomaly_detection.dataset.domain import Datetime
from printer_anomaly_detection.preprocessing.data import Outcome


def load_info(path: Path):
    path = Path(path)
    with open(path / 'info.json') as f:
        info = json.load(f)
    return info

def get_recording_time(path: Path) -> Datetime:
    path = Path(path)
    info = load_info(path)
    return Datetime(datetime.fromisoformat(info['time']))

def get_audio_duration(path: Path) -> float:
    return librosa.get_duration(filename=path.as_posix())

def get_outcome(path: Path) -> Outcome:
    path = Path(path)
    info = load_info(path)
    return Outcome(info['outcome'])

def _windowed_sft(audio: np.ndarray, size: int, step_size: int) -> Iterator[np.ndarray]:
    n_fft = (size - 1) * 2
    hop_length = n_fft // 4

    for i in range(0, audio.shape[0], step_size):
        yield np.abs(librosa.stft(audio[i:i+(size * hop_length - 1)], n_fft=n_fft))


def get_audio_dataset_files(after: Datetime, before: Datetime, outcomes: Set[Outcome]) -> Iterator[Path]:
    for print_path in datasets_path.iterdir():
        if print_path.is_file():
            continue
        
        time = get_recording_time(print_path)
        if time < after or time > before:
            continue

        outcome = get_outcome(print_path)
        if outcome not in outcomes:
            continue

        yield print_path / 'audio.mp4'

@cache
def load_audio_file(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path.as_posix())
    return audio


def load_audio_dataset(after: Datetime, before: Datetime, window_size: int, step_size: int, outcomes: Set[Outcome] = frozenset(Outcome.SUCCESS)) -> Iterator[np.ndarray]:
    for audio_path in get_audio_dataset_files(after, before, outcomes):
        audio = load_audio_file(audio_path)
        for window in _windowed_sft(audio, window_size, step_size):
            yield window
@dataclass
class Upgrade:
    what: str
    when: Datetime
    effects: str = 'audio;video'


datasets_path = Path('./datasets/prints/')
size = (256,)*2

with open(datasets_path / 'upgrades.csv', 'r', encoding='utf-8') as f:
    upgrades = list(sorted(dataclass_csv.DataclassReader(f, Upgrade), key= lambda u: u.when))

before = Datetime.min
for upgrade in upgrades + [Upgrade('nothing', Datetime.max)]:
    if 'audio' not in upgrade.effects:
        continue
    after = upgrade.when
    success_files = list(get_audio_dataset_files(before, after, {Outcome.SUCCESS}))
    number_of_successes = len(success_files)
    success_duration = sum(map(get_audio_duration, success_files))
    number_of_failures = len(list(get_audio_dataset_files(before, after, {Outcome.FAILURE})))
    print(before, after, f'then_upgraded={upgrade.what} +={number_of_successes} -={number_of_failures} +_duration={success_duration}')
    before = after

images = 0
for window in load_audio_dataset(after = Datetime.min, before = Datetime.max, window_size=size, step_size=window_size//3):
    if window.shape != (1025, 1025):
        continue
    break

def main():

    pass