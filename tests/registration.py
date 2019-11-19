import os

from imports.data_generators.image_reg_mask_generator import ImageRegMaskGenerator
from tests.tools.test import Test
from imports.registration import Registration
from imports.settings_parser import SettingsParser
import numpy as np
import cv2


class RegistrationTester:
    def __init__(self):
        with Test('Parser settings loading'):
            parser = SettingsParser(os.path.join(os.path.dirname(__file__), '..', 'settings.json'))

        # with Test('Creating and loading registration database'):
        #     reg = Registration(parser.images_path, parser.masks_path, parser.descriptor_path,
        #                        **parser.registration_args)
        #
        # with Test('Segment image from array'):
        #     filenames = os.listdir(parser.images_path)
        #     filename = filenames[np.random.randint(0, len(filenames))]
        #     path = os.path.join(parser.images_path, filename)
        #     reg.segment(image_filename=path)
        #
        # with Test('Segment image from path'):
        #     reg.segment(image=cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        #
        # with Test('Segment image with not full image list'):
        #     reg = Registration(parser.images_path, parser.masks_path, parser.descriptor_path,
        #                        **parser.registration_args, images_list=filenames[:10])
        #     reg.segment(image=cv2.imread(path, cv2.IMREAD_GRAYSCALE))

        with Test('Registration generator init'):
            gen = ImageRegMaskGenerator(parser.images_path, parser.masks_path, parser.reg_path, **parser.generator_args,
                                        **parser.registration_args)

        with Test('Registrtion generator get'):
            gen_fun = gen.train_generator()
            for i in range(gen.train_steps()):
                imgs, masks = next(gen_fun)
                #imgs[0,:,:,2] = imgs[0,:,:,3]
                #cv2.imshow('', imgs[0,:,:,:3])
                #cv2.waitKey(0)
                assert imgs.shape[3] == 4
                assert masks.shape[3] == 1
