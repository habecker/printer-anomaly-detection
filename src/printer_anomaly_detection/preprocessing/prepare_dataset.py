import argparse
import dataclasses
import glob
import gzip
import json
import shutil
from datetime import datetime
from enum import Enum, auto
from itertools import count
from pathlib import Path
from typing import List

import pandas as pd
from dataclass_csv import DataclassWriter
from py2g_utils.mean import MeanMetrics
from py2g_utils.progress import ProgressMetrics
from py2g_utils.time import TimeMetrics

from printer_anomaly_detection.dataset import *

from .data import *


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocessing Script"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        metavar='DIRECTORY',
        required=True,
        help="Path to the data folder containing all data"
    )
    parser.add_argument(
        "--datasets_folder",
        type=str,
        required=True,
        help="Path for saving the resulting datasets"
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip existing datasets'
    )

    args = parser.parse_args()

    data_path = Path(args.data_folder)
    datasets_path = Path(args.datasets_folder)

    print_paths = list(filter(lambda path: path.is_dir(), data_path.glob('*')))


    number_of_steps = len(print_paths)
    time = TimeMetrics()
    progress = ProgressMetrics()
    progress.start(number_of_steps)

    for print_path in print_paths:
        time.start('dataset')
        files = list(filter(lambda p: p.is_file(), print_path.glob('*')))

        dataset_name = print_path.name
        dataset_path = datasets_path / dataset_name

        if dataset_path.exists() and args.skip_existing:
            continue

        serial_data_files = list(filter(lambda p: p.suffixes == ['.tsv', '.gz'], files))
        klipper_files = list(filter(lambda p: p.suffixes == ['.json', '.gz'], files))
        mp4_files = list(filter(lambda p: p.suffixes == ['.mp4'], files))
        failure_files = list(filter(lambda p: p.name.lower() in ['failure', 'failed'], files))
        remarks_files = list(filter(lambda p: p.name.lower() in ['remarks', 'remarks.md'], files))

        remarks = None
        if (len(remarks_files) > 0):
            with open(remarks_files[0], 'r') as f:
                remarks = f.read()

        assert len(failure_files) <= 1
        assert len(remarks_files) <= 1
        assert len(serial_data_files) == 1
        assert len(klipper_files) == 1
        assert len(mp4_files) == 1


        serial_data_file: Path = serial_data_files.pop()
        klipper_file: Path = klipper_files.pop()
        mp4_file: Path = mp4_files.pop()


        accelerometer, magnetometer, gyroscope, gas = load_serial_file(serial_data_file)

        start_time, heater_target, heater_power, heater_temperature, \
        mcu,\
        extruder_target, extruder_power, extruder_temperature, \
        position_sensor, velocity_sensor, extruder_velocity_sensor, \
        position_target, velocity_target = load_klipper_data(klipper_file)

        everything = zip([  'accelerometer',
                            'magnetometer',
                            'gyroscope',
                            'gas',
                            'heater_target',
                            'heater_power',
                            'heater_temperature',
                            'mcu',
                            'extruder_target',
                            'extruder_power',
                            'extruder_temperature',
                            'position_sensor',
                            'velocity_sensor',
                            'extruder_velocity_sensor',
                            'position_target',
                            'velocity_target'], [
                            accelerometer,
                            magnetometer,
                            gyroscope,
                            gas,
                            heater_target,
                            heater_power,
                            heater_temperature,
                            mcu,
                            extruder_target,
                            extruder_power,
                            extruder_temperature,
                            position_sensor,
                            velocity_sensor,
                            extruder_velocity_sensor,
                            position_target,
                            velocity_target], [
                            IMUSensorData,
                            IMUSensorData,
                            IMUSensorData,
                            GasSensorData,
                            TargetData,
                            PowerSensorData,
                            TemperatureSensorData,
                            MCUData,
                            TargetData,
                            PowerSensorData,
                            TemperatureSensorData,
                            PositionSensorData,
                            VelocitySensorData,
                            VelocitySensorData,
                            PositionTargetData,
                            VelocityTargetData
                            ])

        dataset_path.mkdir(
                exist_ok=True,
                parents=True
            )


        dataset_info = DataInfo(
            name = dataset_name,
            outcome = Outcome.FAILURE if len(failure_files) > 0 else Outcome.SUCCESS,
            remarks=remarks,
            time = start_time.isoformat(),
            video_time = determine_video_time(mp4_file).isoformat()
        )

        with open(dataset_path / 'info.json', 'w', encoding='utf-8') as f:
            json.dump(dataclasses.asdict(dataset_info), f)

        for name, sensor, cls in everything:
            with gzip.open(dataset_path / f'{name}.csv.gz', 'wt', encoding='utf-8') as f:
                writer = DataclassWriter(f, list(sensor), cls)
                writer.write()

        #shutil.copyfile(mp4_file, dataset_path / 'video.mp4')
        extract_video(mp4_file, dataset_path / 'video.mp4')
        extract_audio(mp4_file, dataset_path / 'audio.mp4')
        time.end('dataset')
        progress.step()
        progress.toConsole(main='step', every=1, other=[time])
