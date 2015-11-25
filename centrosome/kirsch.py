import numpy
import scipy.ndimage.filters


def kirsch(image):
    convolution_mask = [5, -3, -3, -3, -3, -3, 5, 5]

    derivatives = numpy.zeros(image.shape)

    for _ in range(len(convolution_mask)):
        derivatives = numpy.maximum(derivatives, scipy.ndimage.filters.convolve(image, numpy.array(convolution_mask[:4] + [0] + convolution_mask[4:]).reshape(3, 3)))

        convolution_mask = convolution_mask[-1:] + convolution_mask[:-1]

    return derivatives
