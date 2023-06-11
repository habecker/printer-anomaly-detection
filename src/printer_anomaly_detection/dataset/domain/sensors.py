from dataclasses import dataclass
from datetime import datetime
from typing import Any

from printer_anomaly_detection.dataset.domain import Datetime, SensorType, SensorData

@dataclass(kw_only=True)
class IMUSensorData(SensorData):
    sensor_type: SensorType
    x: float
    y: float
    z: float

    @classmethod
    def load(cls, sensor_type: SensorType, datetime: datetime, data: list[Any]) -> "IMUSensorData":
        return IMUSensorData(
            datetime=Datetime(datetime),
            sensor_type=sensor_type,
            x=data[0],
            y=data[1],
            z=data[2]
        )

@dataclass(kw_only=True)
class GasSensorData(SensorData):
    sensor_type: SensorType
    value: int

    @classmethod
    def load(cls, sensor_type: SensorType, datetime: datetime, data: list[Any]) -> "GasSensorData":
        return GasSensorData(
            datetime=Datetime(datetime),
            sensor_type=sensor_type,
            value=data[0]
        )
