
import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Type
from dataclass_csv import DataclassReader
import pandas as pd

from py2g_utils.mean import MeanMetrics
from py2g_utils.progress import ProgressMetrics
from py2g_utils.time import TimeMetrics

from printer_anomaly_detection.dataset.domain import Sensor, SensorData
from printer_anomaly_detection.dataset.domain.printer import *
from printer_anomaly_detection.dataset.domain.sensors import *
from .interpolate import interpolators


sensor_filenames: Dict[Sensor, str] = {
    Sensor.EXTRUDER_POWER: 'extruder_power.csv.gz',
    Sensor.EXTRUDER_TARGET: 'extruder_target.csv.gz',
    Sensor.EXTRUDER_TEMPERATURE: 'extruder_temperature.csv.gz',
    Sensor.EXTRUDER_VELOCITY: 'extruder_velocity_sensor.csv.gz',
    Sensor.GAS: 'gas.csv.gz',
    Sensor.HEATER_POWER: 'heater_power.csv.gz',
    Sensor.HEATER_TARGET: 'heater_target.csv.gz',
    Sensor.HEATER_TEMPERATURE: 'heater_temperature.csv.gz',
    Sensor.ACCELEROMETER: 'accelerometer.csv.gz',
    Sensor.MAGNETOMETER: 'magnetometer.csv.gz',
    Sensor.GYROSCOPE: 'gyroscope.csv.gz',
    Sensor.MCU: 'mcu.csv.gz',
    Sensor.POSITION: 'position_sensor.csv.gz',
    Sensor.POSITION_TARGET: 'position_target.csv.gz',
    Sensor.VELOCITY: 'velocity_sensor.csv.gz',
    Sensor.VELOCITY_TARGET: 'velocity_target.csv.gz',
}


sensor_datatypes: Dict[Sensor, Type[SensorData]] = {
    Sensor.EXTRUDER_POWER: PowerSensorData,
    Sensor.EXTRUDER_TARGET: TargetData,
    Sensor.EXTRUDER_TEMPERATURE: TemperatureSensorData,
    Sensor.EXTRUDER_VELOCITY: VelocitySensorData,
    Sensor.GAS: GasSensorData,
    Sensor.HEATER_POWER: PowerSensorData,
    Sensor.HEATER_TARGET: TargetData,
    Sensor.HEATER_TEMPERATURE: TemperatureSensorData,
    Sensor.ACCELEROMETER: IMUSensorData,
    Sensor.MAGNETOMETER: IMUSensorData,
    Sensor.GYROSCOPE: IMUSensorData,
    Sensor.MCU: MCUData,
    Sensor.POSITION: PositionSensorData,
    Sensor.POSITION_TARGET: PositionTargetData,
    Sensor.VELOCITY: VelocitySensorData,
    Sensor.VELOCITY_TARGET: VelocityTargetData,
}

def read_csv(path: Path, dataclass_type: Type):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return list(DataclassReader(f, dataclass_type))


def calculate_availability_time(data: Dict[Sensor, List[SensorData]]) -> Tuple[float, float]:
    """Calculate the time in seconds where all sensors are available. Target sensors are ignored because they're artificially set and stay 'forever'."""
    availability_from = max(
        sensor_data[0].datetime for sensor, sensor_data in data.items() if sensor.has_frequent_updates
    )

    availability_to = min(
        sensor_data[-1].datetime for sensor, sensor_data in data.items() if sensor.has_frequent_updates
    )

    start_time = min(
        sensor_data[0].datetime for _, sensor_data in data.items()
    )

    return start_time.ceil(), availability_from.ceil(), availability_to.floor()

from typing import List
from printer_anomaly_detection.dataset.domain import InterpolationType, Sensor

class DatasetException(Exception): ...

class DatasetBuilder:
    def __init__(self, dataset_name: str, rate: int = 1, offset: int = 0) -> 'DatasetBuilder':
        self.dataset_name = dataset_name
        self.sensors: List[Sensor] = []
        self.interpolations: Dict[Sensor, InterpolationType] = {}
        self.rate: int = rate
        self.offset: int = offset

    def with_sensors(self, sensors: List[Sensor], interpolation: InterpolationType = InterpolationType.CONSTANT) -> 'DatasetBuilder':
        for sensor in sensors:
            self.with_sensor(sensor, interpolation)
        return self

    def with_sensor(self, sensor: Sensor, interpolation: InterpolationType = InterpolationType.CONSTANT) -> 'DatasetBuilder':
        self.sensors.append(sensor)
        self.interpolations[sensor] = interpolation
        return self

    def build(self, dataset_path: Path) -> Tuple[Set[Sensor], pd.DataFrame]:
        # read data
        data: Dict[Sensor, List[SensorData]] = {
            sensor: read_csv(dataset_path / sensor_filenames[sensor], sensor_datatypes[sensor])
            for sensor in self.sensors
        }

        # check sensor availabilities
        missing_sensors = set(filter(lambda sensor: len(data[sensor]) == 0, data.keys()))

        if len(missing_sensors) > 0 and any(filter(lambda sensor: not sensor.is_target, missing_sensors)):
            raise DatasetException(f'No data for non-target sensors {missing_sensors} found!')
        
        # remove empty sensors
        data = {sensor: sensor_data for sensor, sensor_data in data.items() if sensor not in missing_sensors}

        # calculate availability time
        start, availability_from, availability_to = calculate_availability_time(data)
        offset_time = start + self.offset

        start = max(offset_time, availability_from)
        
        # set end time to availability_to as after that the data is not available anymore
        end = availability_to

        # interpolate data
        interpolated_data = {
            sensor: interpolators[self.interpolations[sensor]](data[sensor], rate=self.rate, data_type=type(sensor_data[0])).interpolate(start, end) for sensor, sensor_data in data.items()
        }

        # merge together into one dataframe
        dataframes = []
        
        for sensor, sensor_data in interpolated_data.items():
            df = pd.DataFrame(
                sensor_data,
            ).drop(columns=['sensor_type'])
            # convert map object at column datetime
            df['datetime'] = df['datetime'].map(lambda x: x.timestamp)
            
            # add prefix to all columns except datetime
            df = df.rename(columns={c: f'{sensor.name.lower()}_{c}' for c in df.columns if c not in ['datetime']})
            dataframes.append(df)
        
        assert len(dataframes) > 0

        df = dataframes.pop(0)
        for other_df in dataframes:
            df = df.merge(other_df, on='datetime')

        return missing_sensors, df

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocessing Script"
    )
    parser.add_argument(
        "--prints_dataset",
        type=str,
        metavar='DIRECTORY',
        required=True,
        help="Path to the 'prints' dataset folder"
    )
    parser.add_argument(
        "--merged_dataset",
        type=str,
        metavar='DIRECTORY',
        required=True,
        help="Path to the 'merged' output dataset folder"
    )
    parser.add_argument(
        "--rate",
        type=int,
        metavar='RATE',
        required=True,
        help="Frequency of the merged dataset in Hz"
    )
    parser.add_argument(
        "--skip_existing",
        required=False,
        default=False,
        help="Skip existing merged datasets",
        action='store_true'
    )
    args = parser.parse_args()

    prints_path = Path(args.prints_dataset)
    merged_path = Path(args.merged_dataset)

    print_paths = list(filter(lambda path: path.is_dir(), prints_path.glob('*')))

    number_of_steps = len(print_paths)
    time = TimeMetrics()
    progress = ProgressMetrics()
    progress.start(number_of_steps)

    merged_path.mkdir(parents=True, exist_ok=True)

    meta_path = merged_path / 'meta.json'

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'created_at': datetime.now().isoformat(),
            'name': merged_path.name,
            'rate': args.rate,
        }, f)

    with open(prints_path / 'offsets.csv', 'r', encoding='utf-8') as f:
        offsets = list(csv.DictReader(f))
        offsets = {
            offset['name']: int(offset['offset_seconds']) for offset in offsets
        }

    for print_path in print_paths:
        time.start('dataset')

        output_path = merged_path / print_path.name

        if args.skip_existing and output_path.exists():
            time.end('dataset')
            progress.step()
            progress.toConsole(main='step', every=1, other=[time])
            continue

        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / 'merged.csv.gz'

        try:
            missing_sensors, df = DatasetBuilder(print_path.name, rate=args.rate, offset=offsets.get(print_path.name, 0)) \
                .with_sensors([*Sensor], interpolation = InterpolationType.CONSTANT) \
                .build(dataset_path=print_path)
        except DatasetException as e:
            print(f'Error while building dataset for {print_path.name}: {e}')
            continue

        printed_at = Datetime(df['datetime'][0].item())

        with open(print_path / 'info.json', 'r', encoding='utf-8') as f:
            info_data = json.load(f)

        with open(output_path.parent / 'meta.json', 'w', encoding='utf-8') as f:
            json.dump({
                'printed_at': str(printed_at),
                'duration_in_s': len(df) / float(args.rate),
                'rate': args.rate,
                'missing_sensors': list(map(lambda sensor: sensor.name, missing_sensors)),
                **info_data,
            }, f)

        df.to_csv(output_path, compression='gzip')

        time.end('dataset')
        progress.step()
        progress.toConsole(main='step', every=1, other=[time])
