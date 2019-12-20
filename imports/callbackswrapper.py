"""Callbacks wrapper to JSON serializable"""
from imports.utils.jsonserializable import JSONSerializable


class CallbacksWrapper(JSONSerializable):
    @staticmethod
    def to_json(self):
        """JSON representation of callback"""
        return None

    @staticmethod
    def from_json(json):
        """Get callbacks from JSON"""
        return None
