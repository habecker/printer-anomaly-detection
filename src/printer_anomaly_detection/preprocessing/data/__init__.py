from dataclasses import dataclass
from pathlib import Path
from strenum import StrEnum
from .load_serial import load_serial_file
from .load_klipper import load_klipper_data
from .load_video import determine_video_time, extract_audio, extract_video

class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"

@dataclass
class DataInfo:
    name: str
    outcome: Outcome
    remarks: str
    time: str
    video_time: str

__all__ = ['Outcome', 'DataInfo', 'load_serial_file', 'load_klipper_data', 'determine_video_time', 'extract_audio', 'extract_video']
