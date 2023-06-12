from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Set
from dataclass_csv import dateformat

from strenum import StrEnum

class Datetime:
    """Represents a datetime object without a timezone."""
    def __init__(self, representation: Any) -> None:
        """Creates a new Datetime object from a string, float, Datetime or datetime object."""
        if type(representation) is str:
            self.datetime = datetime.fromisoformat(representation)
            # remove time offset and convert to UTC
            offset = self.datetime.utcoffset() or timedelta()
            self.datetime = self.datetime.replace(tzinfo=None) - offset
        elif type(representation) is float:
            self.datetime = datetime.fromtimestamp(representation)
        elif type(representation) is Datetime:
            self.datetime = representation.datetime.replace()
        elif type(representation) is datetime:
            self.datetime = representation
            # remove time offset and convert to UTC
            offset = self.datetime.utcoffset() or timedelta()
            self.datetime = self.datetime.replace(tzinfo=None) - offset
        else:
            raise TypeError(f"Invalid type for Datetime: {type(representation)}")

    def __le__(self, other: "Datetime") -> bool:
        return self.datetime <= other.datetime

    def __lt__(self, other: "Datetime") -> bool:
        return self.datetime < other.datetime

    def __ge__(self, other: "Datetime") -> bool:
        return self.datetime >= other.datetime

    def __gt__(self, other: "Datetime") -> bool:
        return self.datetime > other.datetime

    def __eq__(self, other: "Datetime") -> bool:
        return self.datetime == other.datetime

    def __sub__(self, other: "Datetime") -> timedelta:
        return self.datetime - other.datetime

    def __add__(self, other: float) -> "Datetime":
        return Datetime(self.datetime + timedelta(seconds=other))

    def floor(self) -> "Datetime":
        """Returns a new Datetime object with the microseconds set to 0."""
        datetime = self.datetime.replace(microsecond=0)
        return Datetime(datetime)

    def ceil(self) -> "Datetime":
        """Returns a new Datetime object with the microseconds set to 0 and the seconds increased by 1."""
        datetime = self.datetime.replace(microsecond=0) + timedelta(seconds=1)
        return Datetime(datetime)

    @property
    def timestamp(self) -> float:
        return self.datetime.timestamp()

    def __str__(self) -> str:
        return self.datetime.isoformat()

    def __repr__(self) -> str:
        return str(self)


class Sensor(Enum):
    EXTRUDER_POWER = auto()
    EXTRUDER_TARGET = auto()
    EXTRUDER_TEMPERATURE = auto()
    EXTRUDER_VELOCITY = auto()
    GAS = auto()
    HEATER_POWER = auto()
    HEATER_TARGET = auto()
    HEATER_TEMPERATURE = auto()
    ACCELEROMETER = auto()
    MAGNETOMETER = auto()
    GYROSCOPE = auto()
    MCU = auto()
    POSITION = auto()
    POSITION_TARGET = auto()
    VELOCITY = auto()
    VELOCITY_TARGET = auto()

    @property
    def has_frequent_updates(self) -> bool:
        """Returns True if the sensor is updated frequently (e.g. every 100ms)."""
        return self in {Sensor.ACCELEROMETER, Sensor.MAGNETOMETER, Sensor.GYROSCOPE, Sensor.EXTRUDER_TEMPERATURE}

    @property
    def is_target(self) -> bool:
        """Returns True if the sensor contains target data."""
        return self in {Sensor.EXTRUDER_TARGET, Sensor.HEATER_TARGET, Sensor.POSITION_TARGET, Sensor.VELOCITY_TARGET}


class SensorType(StrEnum):
    """Enum for the different types of sensors. The values are a short representation of the sensor type - used in the serial communication."""
    ACCELEROMETER = 'A'
    MAGNETOMETER = 'M'
    GYROSCOPE = 'G'
    GAS = 'C',
    HEATER = 'H',
    EXTRUDER = 'E',
    MOTION = 'M'
    MCU = 'X'

    @classmethod
    def imu_types(cls) -> Set["SensorType"]:
        return {SensorType.ACCELEROMETER, SensorType.MAGNETOMETER, SensorType.GYROSCOPE}

class InterpolationType(Enum):
    """Enum for the types of interpolation used when up or downsampling."""
    LINEAR = auto()
    CONSTANT = auto()

@dataclass(kw_only=True)
class SensorData:
    """Super type of all sensor data classes."""
    datetime: Datetime
