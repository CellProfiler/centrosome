import bisect
import h5py
import hashlib
import logging
import numpy
import pickle
import sklearn.decomposition
import sklearn.ensemble

A_CLASS = 'Class'
A_DIGEST = 'MD5Digest'
A_VERSION = "Version"
AA_ADVANCED = "Advanced"
AA_AUTOMATIC = "Automatic"
CLS_CLASSIFIER = "Classifier"
CLS_FILTER = "Filter"
CLS_GROUND_TRUTH = "GroundTruth"
CLS_KERNEL = "Kernel"
CLS_SAMPLING = "Sampling"
DEFAULT_MIN_SAMPLES_PER_LEAF = 10
DEFAULT_N_ESTIMATORS = 25
DEFAULT_N_FEATURES = 100
DEFAULT_RADIUS = 9
DS_COORDS = "Coords"
DS_IMAGE_NUMBER = "ImageNumber"
DS_KERNEL = "Kernel"
G_CLASSIFIERS = "Classifiers"
G_FILTERS = "Filters"
G_IMAGES = "Images"
G_SAMPLING = "Sampling"
G_TRAINING_SET = "TrainingSet"
MODE_CLASSIFY = "Classify"
MODE_TRAIN = "Train"
ROUNDS = [("initial", 100000, 0), ("middle", 75000, 25000), ("final", 50000, 50000)]
SRC_ILASTIK = "Ilastik"
SRC_OBJECTS = "Objects"
USE_DOT = True

logger = logging.getLogger(__name__)

#
# The classifier class is here mostly to simplify the initial development
# and it would get moved out to Centrosome eventually.
#
class Classifier(object):
    '''Represents a classifier stored in an HDF5 file

    The parts:

    Kernel - this is the patch that's taken from the pixel's neighborhood. It
             has the shape, NxM where N is the # of points in the patch and
             M are the indices relative to the pixel. The kernel dimensions
             are C, T, Z, Y, X.
    TrainingSet - the ground truth data. Each label has a corresponding
             dataset stored using the name of the class. The dataset has the
             shape, S x N, where N is the # of points in the patch and S is
             the # of samples for that class.
    Filters - the filters applied to the kernel to derive features. Each
              filter is a vector of length N.
    Classifier - the classifier is pickled after being trained and is stored
                 in a dataset.
    '''
    version = 1
    def __init__(self, path, mode, classifier_path = None):
        '''Either create or load the classifier from a file

        path - path to the file
        mode - "r" for read-only, "w" for overwrite (new file) or "a" for
               read-write access
        classifier_path - path to the sub-groups within the HDF file, defaults
                          to the root.
        '''
        self.f = h5py.File(path, mode)
        self.root = \
            self.f if classifier_path is None else self.f[classifier_path]
        if mode == "w":
            self.f.attrs[A_VERSION] = self.version
            self.g_training_set = self.root.create_group(G_TRAINING_SET)
            self.g_filters = self.root.create_group(G_FILTERS)
            self.g_sampling = self.root.create_group(G_SAMPLING)
            self.g_classifiers = self.root.create_group(G_CLASSIFIERS)
            self.g_images = self.root.create_group(G_IMAGES)
        else:
            self.g_training_set = self.root[G_TRAINING_SET]
            self.g_filters = self.root[G_FILTERS]
            self.g_sampling = self.root[G_SAMPLING]
            self.g_classifiers = self.root[G_CLASSIFIERS]
            self.g_images = self.root[G_IMAGES]
        self.classifier_cache = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.f.flush()
        del self.root
        del self.g_training_set
        del self.g_filters
        del self.g_classifiers
        del self.g_sampling
        self.f.close()
        del self.f

    def random_state(self, extra = ''):
        '''Return a random state based on the ground truth sampled

        '''
        if A_DIGEST not in self.g_training_set.attrs:
            md5 = hashlib.md5()
            for class_name in self.get_class_names():
                md5.update(self.get_ground_truth(class_name).value.data)
            self.g_training_set.attrs[A_DIGEST] = md5.hexdigest()
        return numpy.random.RandomState(numpy.frombuffer(
            self.g_training_set.attrs[A_DIGEST] + extra, numpy.uint8))

    @staticmethod
    def get_instances(group, class_name):
        '''Get the keys for a group for objects of a given class'''
        return sorted([k for k in group.keys()
                       if group[k].attrs[A_CLASS] == class_name])

    def set_kernel(self, kernel):
        '''Set the kernel used to sample pixels from a neighborhood

        kernel - an N x 5 matrix of N offsets by 5 dimensions (C, T, Z, Y, X)
        '''
        if DS_KERNEL in self.root.keys():
            del self.root[DS_KERNEL]
        ds = self.root.create_dataset(DS_KERNEL, data=kernel)
        ds.attrs[A_CLASS] = CLS_KERNEL

    def get_kernel(self):
        return self.root[DS_KERNEL][:]

    def add_class(self, class_name):
        ds = self.g_training_set.create_dataset(
            class_name,
            shape = (0, 6),
            dtype = numpy.int32,
            chunks = (4096, 6),
            maxshape = (None, 6))
        ds.attrs[A_CLASS] = CLS_GROUND_TRUTH

    def get_class_names(self):
        '''Get the names of the classifier's classes'''
        return self.get_instances(self.g_training_set, CLS_GROUND_TRUTH)

    def get_ground_truth(self, class_name):
        '''Get the ground truth for a class

        class_name - the name of the class

        returns an S X 6 where the first index is the image number and
        the remaining are the coordinates of the GT pixel in C, T, Z, Y, X form
        '''
        return self.g_training_set[class_name]

    @property
    def gt_chunk_size(self):
        '''The size of a chunk of ground truth that fits in memory'''
        kernel_size = self.get_kernel().shape[0]
        chunk_size = int(50*1000*1000 / kernel_size)
        return chunk_size

    @property
    def pix_chunk_size(self):
        return 1000 * 1000

    def add_image(self, image, image_number):
        image_number = str(image_number)
        if image_number in self.g_images.keys():
            del self.g_images[image_number]
        self.g_images.create_dataset(image_number, data = image)

    def get_image(self, image_number):
        image_number = str(image_number)
        return self.g_images[image_number].value

    def add_ground_truth(self, class_name, image_number, coordinates):
        '''Add ground truth to a class

        class_name - name of the class
        image_number - the image number as reported in add_image
        pixels - an S x 5 matrix of S samples and 5 pixel coordinates
        '''
        coordinates = numpy.column_stack(
            (numpy.ones(coordinates.shape[0], coordinates.dtype) * image_number,
             coordinates))
        ds = self.get_ground_truth(class_name)
        ds_idx = ds.shape[0]
        ds.resize(ds_idx + coordinates.shape[0], axis = 0)
        ds[ds_idx:] = coordinates

        if A_DIGEST in self.g_training_set.attrs:
            del self.g_training_set.attrs[A_DIGEST]

    def get_samples(self, image, pixels):
        '''Extract samples from an image at given pixels

        image - a C, T, Z, Y, X image

        pixels - an S x 5 matrix where the columns are the C, T, Z, Y, X
                 coordinates and the rows are the samples to collect

        returns an S x N matrix where N is the size of the kernel
        '''
        kernel = self.get_kernel()[numpy.newaxis, :, :]
        coords = pixels[:, numpy.newaxis, :] + kernel
        #
        # Boundary reflection
        #
        coords[coords < 0] = numpy.abs(coords[coords < 0])
        for i, axis_size in enumerate(image.shape):
            mask = coords[:, :, i] >= axis_size
            coords[mask, i] = axis_size * 2 - coords[mask, i] - 1
        #
        # Samples
        #
        samples = image[coords[:, :, 0],
                        coords[:, :, 1],
                        coords[:, :, 2],
                        coords[:, :, 3],
                        coords[:, :, 4]]
        return samples

    def add_sampling(self, sampling_name, d_index):
        '''Add a sampling of the ground truth

        sampling_name - a name for the sampling
        d_index - a dictionary of indices. The key is the class name and
                  the value is a vector of indices into the class's ground truth
        '''
        if sampling_name in self.g_sampling.keys():
            del self.g_sampling[sampling_name]
        g = self.g_sampling.create_group(sampling_name)
        for k, v in d_index.iteritems():
            g.create_dataset(k, data=v)

    def sample(self, sampling_name):
        '''Return sample and vector of class indexes

        sampling_name - the name of the sampling

        Returns a sample which is S x N and vector of length S which is
        composed of indexes into the class names returned by get_class_names.
        S is the length of the sum of all samples in the sampling
        '''
        g = self.g_sampling[sampling_name]
        samples = []
        classes = []
        for idx, class_name in enumerate(self.get_class_names()):
            if class_name in g.keys():
                sampling = g[class_name][:]
                classes.append(numpy.ones(len(sampling), numpy.uint8) * idx)
                gt = self.get_ground_truth(class_name)
                #
                # h5py datasets are not addressable via an array of indices
                # in the way that numpy arrays are. A mask of the array elements
                # to be processed is handled by a loop through each selected
                # element; it takes ~ hours to process arrays of our size. The
                # ground truth may be too large to bring into memory as a
                # Numpy array.
                #
                # We sort the sampling indices and then process in chunks.
                #
                if len(gt) == len(sampling):
                    logger.debug("Extracting %d samples from %s" %
                                 (len(sampling), class_name))
                    samples.append(gt[:])
                else:
                    chunk_size = self.pix_chunk_size
                    sampling.sort()
                    sindx = 0
                    for gtidx in range(0, len(gt), chunk_size):
                        gtidx_end = min(gtidx+chunk_size, len(gt))
                        if sampling[sindx] >= gtidx_end:
                            continue
                        sindx_end = bisect.bisect_left(
                            sampling[sindx:], gtidx_end) + sindx
                        logger.debug(
                            "Extracting %d samples from %s %d:%d" %
                            (sindx_end-sindx, class_name, gtidx, gtidx_end))
                        samples.append(gt[:][sampling[sindx:sindx_end], :])
                        sindx = sindx_end
                        if sindx >= len(gt):
                            break
        samples = numpy.vstack(samples)
        classes = numpy.hstack(classes)
        #
        # Order by image number.
        #
        order = numpy.lexsort(
            [samples[:, _] for _ in reversed(range(samples.shape[1]))])
        samples = samples[order]
        classes = classes[order]
        counts = numpy.bincount(samples[:, 0])
        image_numbers = numpy.where(counts > 0)[0]
        counts = counts[image_numbers]
        idxs = numpy.hstack([[0], numpy.cumsum(counts)])
        result = []
        for image_number, idx, idx_end in zip(
            image_numbers, idxs[:-1], idxs[1:]):
            image = self.get_image(image_number)
            result.append(self.get_samples(image, samples[idx:idx_end, 1:]))

        return numpy.vstack(result), classes

    def make_filter_bank(self, sampling, classes, filter_bank_name, n_filters,
                         algorithm=None):
        '''Make a filter bank using PCA

        sampling - a sampling of the ground truth
        classes - a vector of the same length as the sampling giving the
                  indexes of the classes of each sample
        filter_bank_name - the name to assign to the filter bank
        n_filters - # of filters to create

        algorithm - an object that can be fitted using algorithm.fit(X, Y)
                    and can transform using algorithm.transform(X). Default
                    is RandomizedPCA.
        '''
        if algorithm is None:
            r = self.random_state(filter_bank_name)
            algorithm = sklearn.decomposition.RandomizedPCA(n_filters,
                                                            random_state = r)
        algorithm.fit(sampling, classes)
        if hasattr(algorithm, "components_"):
            components = algorithm.components_
            if len(components) > n_filters:
                components = components[:n_filters]
            if filter_bank_name in self.g_filters.keys():
                del self.g_filters[filter_bank_name]
            ds = self.g_filters.create_dataset(filter_bank_name,
                                               data = components)
            ds.attrs[A_CLASS] = CLS_FILTER
        else:
            s = pickle.dumps(algorithm)
            ds = self.g_filters.create_dataset(filter_bank_name,
                                               data = s)
            ds.attrs[A_CLASS] = CLS_CLASSIFIER

    def use_filter_bank(self, filter_bank_name, sample):
        '''Transform a sample using a filter bank'''
        ds = self.g_filters[filter_bank_name]
        if ds.attrs[A_CLASS] == CLS_FILTER:
            if USE_DOT:
                result = numpy.dot(sample, ds[:].T)
            else:
                #
                # A dot product... but cluster's np.dot is so XXXXed
                #
                chunk_size = self.gt_chunk_size
                result = []
                for idx in range(0, len(sample), chunk_size):
                    idx_end = min(idx + chunk_size, len(sample))
                    logger.debug("Processing dot product chunk %d:%d of %d" %
                                 (idx, idx_end, len(sample)))
                    result.append(numpy.sum(sample[idx:idx_end, :, numpy.newaxis] *
                                         ds[:].T[numpy.newaxis, :, :], 1))
                result = numpy.vstack(result)
        else:
            algorithm = pickle.loads(ds.value)
            result = algorithm.transform(sample)
        return result

    def fit(self, classifier_name, sample, classes, algorithm=None):
        '''Fit an algorithm to data and save

        classifier_name - save using this name
        sample - S samples x N features
        classes - S class labels indexing into the class names
        algorithm - algorithm to use to train
        '''
        if algorithm is None:
            algorithm = sklearn.ensemble.ExtraTreesClassifier(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf = MIN_SAMPLES_PER_LEAF)
        algorithm.fit(sample, classes)
        s = pickle.dumps(algorithm)
        if classifier_name in self.g_classifiers.keys():
            del self.g_classifiers[classifier_name]
        ds = self.g_classifiers.create_dataset(classifier_name, data = s)
        ds.attrs[A_CLASS] = CLS_CLASSIFIER

    def predict_proba(self, classifier_name, sample):
        if classifier_name not in self.classifier_cache:
            algorithm = pickle.loads(self.g_classifiers[classifier_name].value)
            self.classifier_cache[classifier_name] = algorithm
        else:
            algorithm = self.classifier_cache[classifier_name]
        return algorithm.predict_proba(sample)

    def run_pipeline(self, filter_bank_name, classifier_name, sample):
        filtered = self.use_filter_bank(filter_bank_name, sample)
        return self.predict_proba(classifier_name, filtered)

    def config_final_pipeline(self, filter_bank_name, classifier_name):
        self.root.attrs["FilterBankName"] = filter_bank_name
        self.root.attrs["ClassifierName"] = classifier_name

    def run_final_pipeline(self, sample):
        return self.run_pipeline(self.root.attrs["FilterBankName"],
                                 self.root.attrs["ClassifierName"],
                                 sample)
