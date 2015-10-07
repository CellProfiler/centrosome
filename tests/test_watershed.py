import math
import time
import unittest

import numpy
import scipy.ndimage

from centrosome.watershed import fast_watershed

eps = 1e-12

def diff(a, b):
    if not isinstance(a, numpy.ndarray):
        a = numpy.asarray(a)
    if not isinstance(b, numpy.ndarray):
        b = numpy.asarray(b)
    if (0 in a.shape) and (0 in b.shape):
        return 0.0
    b[a==0]=0
    if (a.dtype in [numpy.complex64, numpy.complex128] or
        b.dtype in [numpy.complex64, numpy.complex128]):
        a = numpy.asarray(a, numpy.complex128)
        b = numpy.asarray(b, numpy.complex128)
        t = ((a.real - b.real)**2).sum() + ((a.imag - b.imag)**2).sum()
    else:
        a = numpy.asarray(a)
        a = a.astype(numpy.float64)
        b = numpy.asarray(b)
        b = b.astype(numpy.float64)
        t = ((a - b)**2).sum()
    return math.sqrt(t)

class TestFastWatershed(unittest.TestCase):
    eight = numpy.ones((3,3),bool)
    def test_watershed01(self):
        "watershed 1"
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], numpy.uint8)
        markers = numpy.array([[ -1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 1, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0]],
                                 numpy.int8)
        out = fast_watershed(data, markers,self.eight)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.failUnless(error < eps)

    def test_watershed02(self):
        "watershed 2"
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], numpy.uint8)
        markers = numpy.array([[ -1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 1, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0]],
                                 numpy.int8)
        out = fast_watershed(data, markers)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1,  1,  1,  1, -1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1, -1,  1,  1,  1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.failUnless(error < eps)

    def test_watershed03(self):
        "watershed 3"
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], numpy.uint8)
        markers = numpy.array([[ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 2, 0, 3, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, -1]],
                                 numpy.int8)
        out = fast_watershed(data, markers)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1,  0,  2,  0,  3,  0, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  0,  2,  0,  3,  0, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.failUnless(error < eps)

    def test_watershed04(self):
        "watershed 4"
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], numpy.uint8)
        markers = numpy.array([[ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 2, 0, 3, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, -1]],
                                 numpy.int8)
        out = fast_watershed(data, markers,self.eight)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1, -1, -1, -1, -1, -1, -1],                      
                      [-1, -1, -1, -1, -1, -1, -1],                      
                      [-1, -1, -1, -1, -1, -1, -1],                      
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.failUnless(error < eps)

    def test_watershed05(self):
        "watershed 5"
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], numpy.uint8)
        markers = numpy.array([[ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 3, 0, 2, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, -1]],
                                 numpy.int8)
        out = fast_watershed(data, markers,self.eight)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.failUnless(error < eps)

    def test_watershed06(self):
        "watershed 6"
        data = numpy.array([[0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], numpy.uint8)
        markers = numpy.array([[ 0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 1, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  -1, 0, 0, 0, 0, 0, 0]],
                                 numpy.int8)
        out = fast_watershed(data, markers,self.eight)
        error = diff([[-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.failUnless(error < eps)

    def test_watershed07(self):
        "A regression test of a competitive case that failed"
        data = numpy.array([[255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
                            [255,255,255,255,255,204,204,204,204,204,204,255,255,255,255,255],
                            [255,255,255,204,204,183,153,153,153,153,183,204,204,255,255,255],
                            [255,255,204,183,153,141,111,103,103,111,141,153,183,204,255,255],
                            [255,255,204,153,111, 94, 72, 52, 52, 72, 94,111,153,204,255,255],
                            [255,255,204,153,111, 72, 39,  1, 1, 39, 72,111,153,204,255,255],
                            [255,255,204,183,141,111, 72, 39, 39, 72,111,141,183,204,255,255],
                            [255,255,255,204,183,141,111, 72, 72,111,141,183,204,255,255,255],
                            [255,255,255,255,204,183,141, 94, 94,141,183,204,255,255,255,255],
                            [255,255,255,255,255,204,153,103,103,153,204,255,255,255,255,255],
                            [255,255,255,255,204,183,141, 94, 94,141,183,204,255,255,255,255],
                            [255,255,255,204,183,141,111, 72, 72,111,141,183,204,255,255,255],
                            [255,255,204,183,141,111, 72, 39, 39, 72,111,141,183,204,255,255],
                            [255,255,204,153,111, 72, 39,  1,  1, 39, 72,111,153,204,255,255],
                            [255,255,204,153,111, 94, 72, 52, 52, 72, 94,111,153,204,255,255],
                            [255,255,204,183,153,141,111,103,103,111,141,153,183,204,255,255],
                            [255,255,255,204,204,183,153,153,153,153,183,204,204,255,255,255],
                            [255,255,255,255,255,204,204,204,204,204,204,255,255,255,255,255],
                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]])
        mask = (data!=255)
        markers = numpy.zeros(data.shape,int)
        markers[6,7] = 1
        markers[14,7] = 2
        out = fast_watershed(data, markers,self.eight,mask=mask)
        #
        # The two objects should be the same size, except possibly for the
        # border region
        #
        size1 = numpy.sum(out==1)
        size2 = numpy.sum(out==2)
        self.assertTrue(abs(size1-size2) <=6)
    
    def test_watershed08(self):
        "The border pixels + an edge are all the same value"
        data = numpy.array([[255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
                            [255,255,255,255,255,204,204,204,204,204,204,255,255,255,255,255],
                            [255,255,255,204,204,183,153,153,153,153,183,204,204,255,255,255],
                            [255,255,204,183,153,141,111,103,103,111,141,153,183,204,255,255],
                            [255,255,204,153,111, 94, 72, 52, 52, 72, 94,111,153,204,255,255],
                            [255,255,204,153,111, 72, 39,  1, 1, 39, 72,111,153,204,255,255],
                            [255,255,204,183,141,111, 72, 39, 39, 72,111,141,183,204,255,255],
                            [255,255,255,204,183,141,111, 72, 72,111,141,183,204,255,255,255],
                            [255,255,255,255,204,183,141, 94, 94,141,183,204,255,255,255,255],
                            [255,255,255,255,255,204,153,141,141,153,204,255,255,255,255,255],
                            [255,255,255,255,204,183,141, 94, 94,141,183,204,255,255,255,255],
                            [255,255,255,204,183,141,111, 72, 72,111,141,183,204,255,255,255],
                            [255,255,204,183,141,111, 72, 39, 39, 72,111,141,183,204,255,255],
                            [255,255,204,153,111, 72, 39,  1,  1, 39, 72,111,153,204,255,255],
                            [255,255,204,153,111, 94, 72, 52, 52, 72, 94,111,153,204,255,255],
                            [255,255,204,183,153,141,111,103,103,111,141,153,183,204,255,255],
                            [255,255,255,204,204,183,153,153,153,153,183,204,204,255,255,255],
                            [255,255,255,255,255,204,204,204,204,204,204,255,255,255,255,255],
                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]])
        mask = (data!=255)
        markers = numpy.zeros(data.shape,int)
        markers[6,7] = 1
        markers[14,7] = 2
        out = fast_watershed(data, markers,self.eight,mask=mask)
        #
        # The two objects should be the same size, except possibly for the
        # border region
        #
        size1 = numpy.sum(out==1)
        size2 = numpy.sum(out==2)
        self.assertTrue(abs(size1-size2) <=6)
    
    def test_watershed09(self):
        """Test on an image of reasonable size
        
        This is here both for timing (does it take forever?) and to
        ensure that the memory constraints are reasonable
        """
        image = numpy.zeros((1000,1000))
        coords = numpy.random.uniform(0,1000,(100,2)).astype(int)
        markers = numpy.zeros((1000,1000),int)
        idx = 1
        for x,y in coords:
            image[x,y] = 1
            markers[x,y] = idx
            idx += 1
        
        image = scipy.ndimage.gaussian_filter(image, 4)
        before = time.clock() 
        fast_watershed(image,markers,self.eight)
        elapsed = time.clock()-before
        print "Fast watershed ran a megapixel image in %f seconds"%(elapsed)
        before = time.clock()
        scipy.ndimage.watershed_ift(image.astype(numpy.uint16), markers, self.eight)
        elapsed = time.clock()-before
        print "Scipy watershed ran a megapixel image in %f seconds"%(elapsed)

    def test_watershed10(self):
        # https://github.com/scikit-image/scikit-image/issues/803
        #
        # Make sure that no point in a level image is farther away
        # from its seed than any other
        #
        image = numpy.zeros((21, 21))
        markers = numpy.zeros((21, 21), int)
        markers[5, 5] = 1
        markers[5, 10] = 2
        markers[10, 5] = 3
        markers[10, 10] = 4

        structure = numpy.array([[False, True, False],
                              [True, True, True],
                              [False, True, False]])
        out = fast_watershed(image, markers, structure)
        i, j = numpy.mgrid[0:21, 0:21]
        d = numpy.dstack(
            [numpy.sqrt((i.astype(float)-i0)**2, (j.astype(float)-j0)**2)
             for i0, j0 in ((5, 5), (5, 10), (10, 5), (10, 10))])
        dmin = numpy.min(d, 2)
        self.assertTrue(numpy.all(d[i, j, out[i, j]-1] == dmin))
