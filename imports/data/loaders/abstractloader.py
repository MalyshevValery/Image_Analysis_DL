"""AbstractLoader class - base for other loaders"""
import abc


class AbstractLoader:
    """
    Abstract loader with methods required by MaskGenerator
    """

    @abc.abstractmethod
    def get_image(self, i):
        """Load image from any source and returns it

        :param i: index
        :return: loaded image
        """
        raise NotImplementedError

    def get_mask(self, i):
        """

        :param i: index
        :return: loaded mask, None if its predicting loader
        """

        raise NotImplementedError
