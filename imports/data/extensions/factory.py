"""Factory of extensions"""
from .abstract import AbstractExtension
from .ignoreregion import IgnoreRegionExtension
from .splitmask import SplitMaskExtension
from .typescale import TypeScaleExtension


def extension_factory(json, apply_to='all') -> AbstractExtension:
    """Creates proper extension from specified JSON and checks its application permission"""
    extension_type = json['type']
    if extension_type == TypeScaleExtension.type():
        extension = TypeScaleExtension.from_json(json)
    elif extension_type == IgnoreRegionExtension.type():
        extension = IgnoreRegionExtension.from_json(json)
    elif extension_type == SplitMaskExtension.type():
        extension = SplitMaskExtension.from_json(json)
    else:
        raise ValueError('Type ' + json['type'] + " is unknown type")

    extension.check_extension(apply_to)
    return extension
