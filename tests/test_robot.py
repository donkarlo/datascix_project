from robotix.robot import Robot
from sensorx.sensor_set import SensorSet
class TestRobot:
    def test__init__(self, sensor_set:SensorSet)->None:
        robot = Robot()