"""test_fastemd.py test the FastEMD library wrapper

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import unittest
import centrosome._fastemd

class TestFastEMD(unittest.TestCase):
    def check(
        self, p, q, c, expected_result,
        expected_flow = None,
        extra_mass_penalty = None,
        flow_type = centrosome._fastemd.EMD_NO_FLOW,
        gd_metric = False):
        if p.dtype == centrosome._fastemd.np.int32:
            fn = centrosome._fastemd.emd_hat_int32
            equal_test = self.assertEqual
            array_equal_test = centrosome._fastemd.np.testing.assert_array_equal
        else:
            self.fail("Unsupported dtype: %s" % repr(p.dtype))

        if flow_type == centrosome._fastemd.EMD_NO_FLOW:
            result = fn(p, q, c,
                        extra_mass_penalty=extra_mass_penalty,
                        flow_type=flow_type,
                        gd_metric=gd_metric)
            equal_test(result, expected_result)
        else:
            result, f = fn(p, q, c,
                        extra_mass_penalty=extra_mass_penalty,
                        flow_type=flow_type,
                        gd_metric=gd_metric)
            equal_test(result, expected_result)
            array_equal_test(f, expected_flow)

    def test_01_01_no_flow(self):
        tests = (
            ([1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             centrosome._fastemd.np.zeros((5, 5), centrosome._fastemd.np.int32), 0),
            ([1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             2 - centrosome._fastemd.np.eye(5, dtype=centrosome._fastemd.np.int32), 5),
            ([1, 2, 3, 4, 5],
             [3, 3, 3, 3, 3],
             [[1, 9, 9, 9, 9],
              [2, 9, 9, 9, 9],
              [9, 9, 9, 9, 2],
              [9, 9, 1, 5, 9],
              [9, 3, 4, 9, 9]],
             1*1+2*2+3*2+(3*5 + 1*1)+(3*3+2*4)),
            ([1, 2, 3, 4, 5],
             [5, 10],
             [[1, 9],
              [5, 9],
              [3, 4],
              [9, 5],
              [9, 6]],
             1*1+2*5+(2*3+1*4)+4*5+5*6),
            ([5, 10],
             [1, 2, 3, 4, 5],
             centrosome._fastemd.np.array([[1, 9],
                       [5, 9],
                       [3, 4],
                       [9, 5],
                       [9, 6]], centrosome._fastemd.np.int32).T,
             1*1+2*5+(2*3+1*4)+4*5+5*6)

        )

        for p, q, c, expected in tests:
            self.check(centrosome._fastemd.np.array(p, centrosome._fastemd.np.int32),
                       centrosome._fastemd.np.array(q, centrosome._fastemd.np.int32),
                       centrosome._fastemd.np.array(c, centrosome._fastemd.np.int32),
                       expected)

    def test_01_02_extra_default(self):
        self.check(
            centrosome._fastemd.np.array([1, 2, 3, 4, 5], centrosome._fastemd.np.int32),
            centrosome._fastemd.np.array([5, 15], centrosome._fastemd.np.int32),
            centrosome._fastemd.np.array([[ 1, 10],
                      [ 5, 10],
                      [ 2,  3],
                      [10,  4],
                      [10,  6]], centrosome._fastemd.np.int32),
             1*1+2*5+(2*2+1*3)+4*4+5*6+5*10)

    def test_01_03_threshold(self):
        self.check(
            centrosome._fastemd.np.array([1, 2, 3, 4, 5], centrosome._fastemd.np.int32),
            centrosome._fastemd.np.array([5, 15], centrosome._fastemd.np.int32),
            centrosome._fastemd.np.array([[ 1, 10],
                      [ 5, 10],
                      [ 2,  3],
                      [10,  4],
                      [10,  6]], centrosome._fastemd.np.int32),
             1*1+2*5+(2*2+1*3)+4*4+5*6+5*6,
             extra_mass_penalty=6)

    def test_02_01_flow(self):
        tests = (
            ([1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             2 - centrosome._fastemd.np.eye(5, dtype=centrosome._fastemd.np.int32), 5,
             centrosome._fastemd.np.eye(5, dtype=centrosome._fastemd.np.int32)),
            ([1, 2, 3, 4, 5],
             [3, 3, 3, 3, 3],
             [[1, 9, 9, 9, 9],
              [2, 9, 9, 9, 9],
              [9, 9, 9, 9, 2],
              [9, 9, 1, 5, 9],
              [9, 3, 4, 9, 9]],
             1*1+2*2+3*2+(3*5 + 1*1)+(3*3+2*4),
             [[1, 0, 0, 0, 0],
              [2, 0, 0, 0, 0],
              [0, 0, 0, 0, 3],
              [0, 0, 1, 3, 0],
              [0, 3, 2, 0, 0]]),
            ([1, 2, 3, 4, 5],
             [5, 10],
             [[1, 9],
              [5, 9],
              [3, 4],
              [9, 5],
              [9, 6]],
             1*1+2*5+(2*3+1*4)+4*5+5*6,
             [[1, 0],
              [2, 0],
              [2, 1],
              [0, 4],
              [0, 5]]),
            ([5, 10],
             [1, 2, 3, 4, 5],
             centrosome._fastemd.np.array([[1, 9],
                       [5, 9],
                       [3, 4],
                       [9, 5],
                       [9, 6]], centrosome._fastemd.np.int32).T,
             1*1+2*5+(2*3+1*4)+4*5+5*6,
             centrosome._fastemd.np.array([[1, 0],
              [2, 0],
              [2, 1],
              [0, 4],
              [0, 5]]).T)

        )

        for p, q, c, expected, expected_flow in tests:
            self.check(centrosome._fastemd.np.array(p, centrosome._fastemd.np.int32),
                       centrosome._fastemd.np.array(q, centrosome._fastemd.np.int32),
                       centrosome._fastemd.np.array(c, centrosome._fastemd.np.int32),
                       expected,
                       expected_flow = centrosome._fastemd.np.array(expected_flow),
                       flow_type=centrosome._fastemd.EMD_WITHOUT_EXTRA_MASS_FLOW)
        
        