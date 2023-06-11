

from datetime import datetime
from typing import Any, List

from printer_anomaly_detection.dataset.domain import SensorData, Sensor, SensorType
from printer_anomaly_detection.dataset.domain.printer import PositionSensorData, PositionTargetData, PowerSensorData, TargetData, TemperatureSensorData, VelocitySensorData
from printer_anomaly_detection.dataset.domain.sensors import GasSensorData, IMUSensorData


# def load(datetime: datetime, sensor: Sensor, sensor_type: SensorType, data: List[Any]) -> SensorData:
#     # match case for each value of the Sensor enum with pass
#     match sensor:
#         case Sensor.EXTRUDER_POWER:
#             assert sensor_type == SensorType.EXTRUDER
#             return PowerSensorData.load(datetime=datetime, sensor_type=SensorType.EXTRUDER, data=data)
#         case Sensor.EXTRUDER_TARGET:
#             assert sensor_type == SensorType.EXTRUDER
#             return TargetData.load(datetime=datetime, sensor_type=SensorType.EXTRUDER, data=data)
#         case Sensor.EXTRUDER_TEMPERATURE:
#             assert sensor_type == SensorType.EXTRUDER
#             return TemperatureSensorData.load(datetime=datetime, sensor_type=SensorType.EXTRUDER, data=data)
#         case Sensor.EXTRUDER_VELOCITY:
#             assert sensor_type == SensorType.EXTRUDER
#             return VelocitySensorData.load(datetime=datetime, sensor_type=SensorType.EXTRUDER, data=data)
#         case Sensor.GAS:
#             assert sensor_type == SensorType.GAS
#             return GasSensorData.load(datetime=datetime, sensor_type=SensorType.GAS, data=data)
#         case Sensor.HEATER_POWER:
#             assert sensor_type == SensorType.HEATER
#             return PowerSensorData.load(datetime=datetime, sensor_type=SensorType.HEATER, data=data)
#         case Sensor.HEATER_TARGET:
#             assert sensor_type == SensorType.HEATER
#             return TargetData.load(datetime=datetime, sensor_type=SensorType.HEATER, data=data)
#         case Sensor.HEATER_TEMPERATURE:
#             assert sensor_type == SensorType.HEATER
#             return TemperatureSensorData.load(datetime=datetime, sensor_type=SensorType.HEATER, data=data)
#         case Sensor.IMU:
#             assert sensor_type in SensorType.imu_types()

#             if sensor_type in SensorType.ACCELEROMETER:
#                 return IMUSensorData.load(SensorType.ACCELEROMETER, datetime, data)
#             elif sensor_type in SensorType.MAGNETOMETER:
#                 return IMUSensorData.load(SensorType.MAGNETOMETER, datetime, data)
#             elif sensor_type in SensorType.GYROSCOPE:
#                 return IMUSensorData.load(SensorType.GYROSCOPE, datetime, data)
#             else:
#                 raise NotImplementedError(f"Sensor type {sensor_type} not implemented for IMU")
#         case Sensor.MCU:
#             assert sensor_type == SensorType.MCU
#             return IMUSensorData.load(SensorType.MCU, datetime, data)
#         case Sensor.POSITION:
#             assert sensor_type == SensorType.MOTION
#             return PositionSensorData.load(datetime=datetime, sensor_type=SensorType.MOTION, data=data)
#         case Sensor.POSITION_TARGET:
#             assert sensor_type == SensorType.MOTION
#             return PositionTargetData.load(datetime=datetime, sensor_type=SensorType.MOTION, data=data)
#         case Sensor.VELOCITY:
#             assert sensor_type == SensorType.MOTION
#             return VelocitySensorData.load(datetime=datetime, sensor_type=SensorType.MOTION, data=data)
#         case Sensor.VELOCITY_TARGET:
#             assert sensor_type == SensorType.MOTION
#             return PositionTargetData.load(datetime=datetime, sensor_type=SensorType.MOTION, data=data)
