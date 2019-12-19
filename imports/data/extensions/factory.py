"""Factory of extensions"""
from .ignoreregion import IgnoreRegionExtension
from .typescale import TypeScaleExtension


def extension_factory(json, apply_to='all'):
    """Creates proper extension from specified json"""
    extension_type = json['type']
    if extension_type == TypeScaleExtension.type():
        extension = TypeScaleExtension.from_json(json)
    elif extension_type == IgnoreRegionExtension.type():
        extension = IgnoreRegionExtension.from_json(json)
    else:
        raise ValueError('Type ' + json['type'] + " is unknown type")

    extension.check_extension(apply_to)
    return extension
