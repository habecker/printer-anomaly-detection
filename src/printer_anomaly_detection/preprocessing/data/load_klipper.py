from datetime import timedelta, datetime
import gzip
import json
from pathlib import Path
from typing import Tuple, List, Dict, Set

from printer_anomaly_detection.dataset.domain.printer import *

def get_field(field_name: str, data, printer_start_time, printer_time_offset) -> List[Tuple[datetime, Dict]]:
    def _f(row):
        if 'q' not in row or row['q'] != 'status':
            return False
        return field_name in row['params']['status']

    def _m(row):
        tdelta = timedelta(seconds=(row['params']['eventtime'] + printer_time_offset))
        return printer_start_time + tdelta, row['params']['status'][field_name]
    
    return list(map(_m, filter(_f, data)))

def _load_klipper_json(path: Path) -> List[Dict]:
    with gzip.open(path, 'rb') as f:
        data = f.read()
        data = data.split('\x03'.encode('ascii'))
        data = filter(lambda i: len(i) > 0, data)
        data = [json.loads(i) for i in data]
    return data

def _determine_time_offset(klipper_json_path: Path, data: List[Dict]) -> Tuple[datetime, datetime]:
    prev = 0
    for d in filter(lambda d: 'q' in d and 'eventtime' in d['params'], data):
        eventtime = d['params']['eventtime']
        assert eventtime > prev
    prev = eventtime
    name = klipper_json_path.name[:-len('.json.gz')]
    datetimestr = name[:-len('_klipper')]
    printer_start_time: datetime = datetime.strptime(datetimestr[:-len('000Z')], '%Y-%m-%dT%H%M%S%f')
    printer_time_offset = -next(filter(lambda d: 'q' in d, data))['params']['eventtime']
    return printer_start_time, printer_time_offset

def _load_heater_data(data, printer_start_time, printer_time_offset):
    heater_data = get_field('heater_bed', data, printer_start_time, printer_time_offset)
    heater_target = map(
      lambda d: TargetData(sensor_type=SensorType.HEATER, datetime=Datetime(d[0]), target=d[1]['target']),
      filter(lambda d: 'target' in d[1], heater_data)
    )
    heater_power = map(
      lambda d: PowerSensorData(sensor_type=SensorType.HEATER, datetime=Datetime(d[0]), power=d[1]['power']),
      filter(lambda d: 'power' in d[1], heater_data)
    )
    heater_temperature = map(
      lambda d: TemperatureSensorData(sensor_type=SensorType.HEATER, datetime=Datetime(d[0]), temperature=d[1]['temperature']),
      filter(lambda d: 'temperature' in d[1], heater_data)
    )
    return heater_target, heater_power, heater_temperature

def _load_mcu_data(data, printer_start_time, printer_time_offset):
    mcu_data = get_field('mcu', data, printer_start_time, printer_time_offset)
    mcu = map(
    lambda d: MCUData(
        sensor_type=SensorType.MCU,
        datetime=Datetime(d[0]),
        upcoming_bytes=d[1]['last_stats']['upcoming_bytes'],
        srtt=d[1]['last_stats']['srtt'],
        bytes_invalid=d[1]['last_stats']['bytes_invalid'],
        send_seq=d[1]['last_stats']['send_seq'],
        retransmit_seq=d[1]['last_stats']['retransmit_seq'],
        rttvar=d[1]['last_stats']['rttvar'],
        rto=d[1]['last_stats']['rto'],
        bytes_retransmit=d[1]['last_stats']['bytes_retransmit'],
        bytes_write=d[1]['last_stats']['bytes_write'],
        ready_bytes=d[1]['last_stats']['ready_bytes'],
        freq=d[1]['last_stats']['freq'],
        receive_seq=d[1]['last_stats']['receive_seq']),
    mcu_data
    )
    return mcu

def _load_extruder_data(data, printer_start_time, printer_time_offset):
    extruder_data = get_field('extruder', data, printer_start_time, printer_time_offset)
    extruder_target = map(
        lambda d: TargetData(sensor_type=SensorType.EXTRUDER, datetime=Datetime(d[0]), target=d[1]['target']),
        filter(lambda d: 'target' in d[1], extruder_data)
    )
    extruder_power = map(
        lambda d: PowerSensorData(sensor_type=SensorType.EXTRUDER, datetime=Datetime(d[0]), power=d[1]['power']),
        filter(lambda d: 'power' in d[1], extruder_data)
    )
    extruder_temperature = map(
        lambda d: TemperatureSensorData(sensor_type=SensorType.EXTRUDER, datetime=Datetime(d[0]), temperature=d[1]['temperature']),
        filter(lambda d: 'temperature' in d[1], extruder_data)
    )
    return extruder_target, extruder_power, extruder_temperature

def _load_motion_report_data(data, printer_start_time, printer_time_offset):
    motion_report_data = get_field('motion_report', data, printer_start_time, printer_time_offset)
    position_sensor = map(
        lambda d: PositionSensorData(
            sensor_type=SensorType.MOTION,
            datetime=Datetime(d[0]),
            x=d[1]['live_position'][0],
            y=d[1]['live_position'][1],
            z=d[1]['live_position'][2],
            e=d[1]['live_position'][3],
        ),
        filter(lambda d: 'live_position' in d[1], motion_report_data)
    )
    velocity_sensor = map(
        lambda d: VelocitySensorData(
            sensor_type=SensorType.MOTION,
            datetime=Datetime(d[0]),
            velocity=d[1]['live_velocity'],
        ),
        filter(lambda d: 'live_velocity' in d[1], motion_report_data)
    )
    extruder_velocity_sensor = map(
        lambda d: VelocitySensorData(
            sensor_type=SensorType.EXTRUDER,
            datetime=Datetime(d[0]),
            velocity=d[1]['live_extruder_velocity'],
        ),
        filter(lambda d: 'live_extruder_velocity' in d[1], motion_report_data)
    )
    return position_sensor, velocity_sensor, extruder_velocity_sensor

def _load_gcode_data(data, printer_start_time, printer_time_offset):
    gcode_data = get_field('gcode_move', data, printer_start_time, printer_time_offset)
    position_target = map(
    lambda d: PositionTargetData(
        sensor_type=SensorType.MOTION,
        datetime=Datetime(d[0]),
        x=d[1]['position'][0],
        y=d[1]['position'][1],
        z=d[1]['position'][2],
        e=d[1]['position'][3],
    ),
    filter(lambda d: 'position' in d[1], gcode_data)
    )
    velocity_target = map(
    lambda d: VelocityTargetData(
        sensor_type=SensorType.MOTION,
        datetime=Datetime(d[0]),
        velocity=d[1]['speed'],
    ),
    filter(lambda d: 'speed' in d[1], gcode_data)
    )
    return position_target, velocity_target


def load_klipper_data(path: Path) -> List[Dict]:
    data = _load_klipper_json(path)
    printer_start_time, printer_time_offset =_determine_time_offset(path, data)

    heater_target, heater_power, heater_temperature = _load_heater_data(data, printer_start_time, printer_time_offset)
    mcu = _load_mcu_data(data, printer_start_time, printer_time_offset)
    extruder_target, extruder_power, extruder_temperature = _load_extruder_data(data, printer_start_time, printer_time_offset)
    position_sensor, velocity_sensor, extruder_velocity_sensor = _load_motion_report_data(data, printer_start_time, printer_time_offset)
    position_target, velocity_target = _load_gcode_data(data, printer_start_time, printer_time_offset)

    return  printer_start_time, \
            heater_target, heater_power, heater_temperature, \
            mcu,\
            extruder_target, extruder_power, extruder_temperature, \
            position_sensor, velocity_sensor, extruder_velocity_sensor, \
            position_target, velocity_target
        