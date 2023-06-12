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

@dataclass(kw_only=True)
class GasSensorData(SensorData):
    sensor_type: SensorType
    value: float
