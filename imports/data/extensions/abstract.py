"""Abstract extension"""
from abc import abstractmethod

from imports.jsonserializable import JSONSerializable


class AbstractExtension(JSONSerializable):
    """Abstract extension class

    Abstract extensions is used in Loader to extend default capabilities of Storage methods. Mostly extensions should be
    used to change data format, normalize or slightly change it. For augmentation please consult albumentations package
    and Sequence, which is used as main source of data for Keras training and test processes.
    """

    @classmethod
    @abstractmethod
    def allowed(cls):
        """To which data types this Extension can be applied

        Currently available places to insert extensions:
            - all
            - image -- is applied to images after loading
            - mask -- is applied to masks after loading
            - save_image -- is applied to loaded images in save_predicted method of Loader class
            - save -- is applied before saving resulting image in save_predicted method of Loader class
        """
        return []

    @classmethod
    @abstractmethod
    def type(cls):
        """Returns string which identifies extension"""
        return None

    @abstractmethod
    def __call__(self, data):
        """Apply transformation to data"""
        return None

    @classmethod
    def check_extension(cls, apply_to):
        """Check if this extension can be applied to apply_to. If it can not it raises an exception

        :param apply_to: label of place to insert extension call in Loader class
        """
        allowed = cls.allowed()
        if apply_to not in allowed and apply_to != allowed and allowed != 'all' and 'all' not in allowed:
            raise ValueError(apply_to + ' is not allowed in ' + cls.type() + ' - ' + cls.allowed())
