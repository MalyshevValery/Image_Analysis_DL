import os
import numpy as np
import cv2
import sklearn.metrics as MT
import pandas as pd
import tempfile


class Registration:
    def __init__(self, images_dir, masks_dir, descriptor_file, num_images=5, n_jobs=1, images_list=None):
        self.num_images = num_images
        self.size = (256, 256)
        self.descriptors = descriptor_file
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.n_jobs = n_jobs

        if images_list is None:
            images = os.listdir(images_dir)
        else:
            images = images_list

        masks = os.listdir(masks_dir)
        self.filenames = pd.Series([f for f in images if f in masks])
        if len(self.filenames) == 0:
            raise Exception('No common names in images and masks directories')

        parameters_path = os.path.join(os.path.dirname(__file__), 'parameters.txt')
        self.run_elastix = "elastix -f {} -m {} -p " + parameters_path \
                           + " -out {} -threads " + str(n_jobs) + " >/dev/null"
        self.run_transformix = "transformix -in {0} -tp {1}/TransformParameters.0.txt -out {1} >/dev/null"

        if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
            raise Exception('One of directories {} or {} does not exist')

        # Read descriptor
        if not os.path.isfile(descriptor_file):
            # Load Database
            self.descriptors = self._precalculate(descriptor_file)
        else:
            self.descriptors = pd.read_csv(descriptor_file, index_col=None)
            filenames = self.descriptors['filename']
            n_matches = len(set(filenames) & set(self.filenames))
            if n_matches != len(self.filenames) or n_matches != len(filenames):
                self.descriptors = self._precalculate(descriptor_file)
            else:
                self.descriptors = self.descriptors.drop(['filename'], axis=1).values

    def simple_descriptor(self, image):
        res = np.concatenate([np.sum(image, axis=0) / self.size[0], np.sum(image, axis=1) / self.size[1]])
        return res / 255

    def _precalculate(self, filename):
        descriptor = []
        for f in self.filenames:
            image = cv2.imread(os.path.join(self.images_dir, f), cv2.IMREAD_GRAYSCALE)
            descriptor.append((self.simple_descriptor(image)))
        df = pd.DataFrame(np.asarray(descriptor))
        df['filename'] = self.filenames
        df.to_csv(filename, index=None)
        return np.asarray(descriptor)

    def segment(self, image=None, image_filename=None):
        if image is None and image_filename is None:
            raise Exception("no arguments")

        if image is None:
            descriptor = self.simple_descriptor(cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE))
        else:
            descriptor = self.simple_descriptor(image)

        distances = MT.pairwise_distances(self.descriptors, [descriptor], metric='correlation')[:, 0]
        indexes = np.argsort(distances)[:self.num_images]
        filenames = self.filenames[indexes]

        if image is not None:
            image_filename = next(tempfile._get_candidate_names()) + '.png'
            cv2.imwrite(image_filename, image)
        masks = []
        e = None
        mask = None
        try:
            for f in filenames:
                moving_image = os.path.join(self.images_dir, f)
                moving_mask = os.path.join(self.masks_dir, f)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    os.system(self.run_elastix.format(image_filename, moving_image, tmp_dir))
                    os.system(self.run_transformix.format(moving_mask, tmp_dir))
                    masks.append(cv2.imread(os.path.join(tmp_dir, 'result.bmp'), cv2.IMREAD_GRAYSCALE))
            mask = np.mean(np.asarray(masks), axis=0)
            mask /= mask.max()
        except Exception as e:
            pass
        finally:
            if image is not None:
                os.remove(image_filename)
        if e is not None:
            raise e
        return mask

    @staticmethod
    def __run_registration(elastix_command, transformix_command):
        pass