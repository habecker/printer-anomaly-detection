from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dataclass_csv import dateformat

from printer_anomaly_detection.dataset import SensorType, SensorData
from printer_anomaly_detection.dataset.domain import Datetime

# COPILOT: Remove load functions from all classes

@dateformat('%Y-%m-%d %H:%M:%S.%f')
@dataclass(kw_only=True)
class TemperatureSensorData(SensorData):
    sensor_type: SensorType
    temperature: float

@dataclass(kw_only=True)
class TargetData(SensorData):
    sensor_type: SensorType
    target: float

@dataclass(kw_only=True)
class PowerSensorData(SensorData):
    sensor_type: SensorType
    power: float

@dataclass(kw_only=True)
class PositionSensorData(SensorData):
    sensor_type: SensorType
    x: float
    y: float
    z: float
    e: float

@dataclass(kw_only=True)
class VelocitySensorData(SensorData):
    sensor_type: SensorType
    velocity: float

@dataclass(kw_only=True)
class PositionTargetData(SensorData):
    sensor_type: SensorType
    x: float
    y: float
    z: float
    e: float

@dataclass(kw_only=True)
class VelocityTargetData(SensorData):
    sensor_type: SensorType
    velocity: float

@dataclass(kw_only=True)
class MCUData(SensorData):
    sensor_type: SensorType
    upcoming_bytes: int
    srtt: float
    bytes_invalid: int
    send_seq: int
    retransmit_seq: int
    rttvar: float
    rto: float
    bytes_retransmit: int
    bytes_write: int
    ready_bytes: int
    freq: int
    receive_seq: int
