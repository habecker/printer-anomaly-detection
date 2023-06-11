import gzip
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from printer_anomaly_detection.dataset.domain import Datetime, SensorType

from printer_anomaly_detection.dataset.domain.sensors import GasSensorData, IMUSensorData

def load_serial_file(serial_tsv_file: Path) -> Tuple[List[IMUSensorData], List[GasSensorData]]:
    sensor_data: List[IMUSensorData | GasSensorData] = list()
    with gzip.open(serial_tsv_file, 'rb') as f:
        for i, line in enumerate(f):
            data = line.decode('ascii')
            data = data.split('\t')
            datetime_ = data[0]
            datetime_ = datetime.fromisoformat(datetime_)

            try:
                sensor_type = SensorType(data[1])
            except ValueError as exc:
                if i > 0:
                    raise exc
                continue

            if sensor_type in SensorType.imu_types():
                sensor_data.append(
                    IMUSensorData(
                        sensor_type=sensor_type,
                        datetime=Datetime(datetime_),
                        x = float(data[2]),
                        y = float(data[3]),
                        z = float(data[4])
                    )
                )
            elif sensor_type == SensorType.GAS:
                sensor_data.append(
                    GasSensorData(
                        sensor_type=sensor_type,
                        datetime=Datetime(datetime_),
                        value=float(data[2])
                    )
                )
            else:
                raise ValueError("Invalid sensor type")
    accelerometer = list(filter(lambda d: isinstance(d, IMUSensorData) and d.sensor_type == SensorType.ACCELEROMETER, sensor_data))
    magnetometer = list(filter(lambda d: isinstance(d, IMUSensorData) and d.sensor_type == SensorType.MAGNETOMETER, sensor_data))
    gyroscope = list(filter(lambda d: isinstance(d, IMUSensorData) and d.sensor_type == SensorType.GYROSCOPE, sensor_data))
    gas = list(filter(lambda d: isinstance(d, GasSensorData), sensor_data))
    return accelerometer, magnetometer, gyroscope, gas
