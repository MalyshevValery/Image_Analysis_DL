import os
import numpy as np
import cv2
import sklearn.metrics as MT
import pandas as pd
import tempfile


class Registration:
    """Class for image registration by elastix"""
    def __init__(self, images_dir, masks_dir, descriptor_file, num_images=5, n_jobs=1, images_list=None,
                 forbid_the_same=True):
        """Constructor

        :param images_dir: dir with images
        :param masks_dir: dir with masks
        :param descriptor_file: file to put descriptors in
        :param num_images: Number of neighbour images which will be averaged to result mask
        :param n_jobs: N-jobs for image registration
        :param images_list: list of database image names - If None class use all images in folder
        :param forbid_the_same: Forbids to use provided image for registration if its in database
        """
        self.num_images = num_images
        self.size = (256, 256)
        self.descriptors = descriptor_file
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.n_jobs = n_jobs
        self.forbid_same = forbid_the_same

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
        """Descriptor of image. Row- and Column- wise sums of image"""
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

    def segment(self, image):
        """Segment image with registration. Image can be either ndarray or string path to file"""
        if isinstance(image,str):
            image_filename = image
            image = None

        if image is None:
            descriptor = self.simple_descriptor(cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE))
        else:
            descriptor = self.simple_descriptor(image)

        distances = MT.pairwise_distances(self.descriptors, [descriptor], metric='correlation')[:, 0]
        indexes = np.argsort(distances)[:self.num_images]
        if self.forbid_same and distances[indexes[0]] < 0.001:
            indexes = np.argsort(distances)[1:self.num_images + 1]
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
                    # TODO: weights according to distances
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
