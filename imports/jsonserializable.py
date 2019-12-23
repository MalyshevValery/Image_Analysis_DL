"""Abstract json serializable class"""
from abc import ABCMeta, abstractmethod


class JSONSerializable(metaclass=ABCMeta):
    """Abstract JSON serializable class"""

    @abstractmethod
    def to_json(self):
        """Return python dict to turn to JSON later"""
        return {}

    @staticmethod
    @abstractmethod
    def from_json(json):
        """Creates object from json"""
        return JSONSerializable()
