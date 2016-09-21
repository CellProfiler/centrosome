"""setup.py - setup to build C modules for SizeIntervalPrecision

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

from distutils.core import setup,Extension
import glob
import os
import sys
is_win = sys.platform.startswith("win")
try:
    from Cython.Distutils import build_ext
    from numpy import get_include
except ImportError:
    import site
    site.addsitedir('../../site-packages')
    from Cython.Distutils import build_ext
    from numpy import get_include

def configuration():
    if is_win:
        extra_compile_args = None
        extra_link_args = ['/MANIFEST']
    else:
        extra_compile_args = ['-O3']
        extra_link_args = None
    extensions = [Extension(name="_sizeIntervalPrecision",
                            sources=["src/_sizeIntervalPrecision.c"],
                            include_dirs=['src']+[get_include()],
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args)
                  ]
    dict = { "name":"SizeIntervalPrecision",
             "description":"SizeIntervalPrecision threshold method for CellProfiler",
             "maintainer":"Petter Ranefall",
             "maintainer_email":"petter.ranefall@it.uu.se",
             "cmdclass": {'build_ext': build_ext},
             "ext_modules": extensions
            }
    return dict

if __name__ == '__main__':
    if '/' in __file__:
        os.chdir(os.path.dirname(__file__))
    setup(**configuration())
    

