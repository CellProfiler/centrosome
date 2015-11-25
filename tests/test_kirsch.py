import centrosome.kirsch
import skimage.io
import skimage.data
import scipy.misc
import numpy


def test_kirsch():
    assert numpy.array_equal(centrosome.kirsch.kirsch(skimage.data.camera()), scipy.misc.imread("tests/resources/kirsch.tiff"))
