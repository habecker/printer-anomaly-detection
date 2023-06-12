from typing import Callable, Dict, Generator, Iterator, List, Optional, Type, TypeVar

from printer_anomaly_detection.dataset.domain import Datetime, InterpolationType, SensorData

T = TypeVar('T', bound=SensorData)
class Interpolator:
    def __init__(self, data: List[T], data_type: Type[T], rate: int) -> 'Interpolator':
        self.data = data
        self.rate = rate
        self.period = 1. / rate
        self.data_type = data_type

    def interpolate(self, start_time: Datetime, end_time: Datetime) -> Iterator[T]:
        raise NotImplementedError()


T = TypeVar('T', bound=SensorData)
class LinearInterpolator(Interpolator):
    """Interpolate data by using the last known value"""
    def __init__(self, data: List[T], rate: int, data_type: Type[T], to_floats: Optional[Callable[[T], List[float]]], from_floats: Optional[Callable[[List[float]], T]]) -> 'Interpolator':
        super().__init__(data, rate, data_type)
    
    def interpolate(self, start_time: Datetime, end_time: Datetime) -> Iterator[T]:
        raise NotImplementedError()
        time = start_time
        while self.data[1].datetime < start_time:
            # Remove all data points that are before the start time so that the first data point is the one that is closest to the start time
            self.data.pop(0)
        while True:
            value = self.data.pop(0)
            while value < time:
                value = self.data.pop(0)
            yield value
            time += self.period



class ConstantInterpolator(Interpolator):
    def __init__(self, data: List[T], rate: int, data_type: Type[T]) -> 'Interpolator':
        super().__init__(data, data_type, rate)
 
    def interpolate(self, start_time: Datetime, end_time: Datetime) -> Iterator[T]:
        time = start_time

        value = None

        while True:
            if time >= end_time:
                break

            if len(self.data) == 0:
                assert value is not None
                yield self.data_type(**{**value.__dict__, 'datetime': time})
                time += self.period
                continue

            value = value or self.data[0]
            # if no more data: keep the last data point
            while len(self.data) > 0 and self.data[0].datetime < time:
                # Remove all data points that are before the current time so that the first data point is the one that is closest before the current time
                value = self.data.pop(0)

            yield self.data_type(**{**value.__dict__, 'datetime': time})

            time += self.period

interpolators: Dict[InterpolationType, Interpolator] = {
    InterpolationType.LINEAR: LinearInterpolator,
    InterpolationType.CONSTANT: ConstantInterpolator,
}
