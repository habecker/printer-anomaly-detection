from datetime import datetime
from pathlib import Path
from ffmpeg import FFmpeg

def determine_video_time(video_path: Path) -> datetime:
    datetimestr = video_path.name[:-len('_camera.mp4')]
    return datetime.strptime(datetimestr[:-len('000Z')], '%Y-%m-%dT%H%M%S%f')

def extract_audio(video_path: Path, target_path: Path):
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(str(video_path))
        .output(
            str(target_path),
            {"codec:a": "copy"},
            map="0:a",
        )
    )
    ffmpeg.execute()

def extract_video(video_path: Path, target_path: Path):
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(str(video_path))
        .output(
            str(target_path),
            {"codec:v": "copy"},
            map="0:v",
        )
    )
    ffmpeg.execute()
