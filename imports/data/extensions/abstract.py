"""Abstract extension"""
from abc import abstractmethod

from imports.utils.jsonserializable import JSONSerializable


class AbstractExtension(JSONSerializable):
    """Abstract extension"""

    @classmethod
    @abstractmethod
    def allowed(cls):
        """To which loader parts it can be applied"""
        return []

    @classmethod
    @abstractmethod
    def type(cls):
        """Returns type of extension"""
        return None

    @abstractmethod
    def __call__(self, data):
        """Apply transformation to data"""
        return None

    @classmethod
    def check_extension(cls, apply_to):
        """Check if this extension can be applied to apply_to. If it can not it raises an exception

        :param apply_to: where this extension will be applied
        """
        allowed = cls.allowed()
        if apply_to not in allowed and apply_to != allowed and allowed != 'all' and 'all' not in allowed:
            raise ValueError(apply_to + ' is not allowed in ' + cls.type() + ' - ' + cls.allowed())
