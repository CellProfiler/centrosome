from distutils.core import setup, Extension
import glob
import sys
import numpy
from Cython.Distutils import build_ext

if sys.platform.startswith('win'):
    extra_compile_args = None

    extra_link_args = ['/MANIFEST']
else:
    extra_compile_args = ['-O3']

    extra_link_args = None

setup(
    name='centrosome',
    version='0.7.0',
    description='',
    long_description='',
    url='',
    author='',
    author_email='',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='',
    packages=[
        'centrosome'
    ],
    install_requires=[
        'Cython',
        'numpy',
        'scipy',
    ],
    extras_require={
        'development': [

        ],
        'test': [

        ]
    },
    cmdclass={
        'build_ext': build_ext
    },
    ext_modules=[
        Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                'centrosome/src',
                numpy.get_include(),
            ],
            name='_cpmorphology',
            sources=[
                'centrosome/src/cpmorphology.c'
            ],
        ),
        Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                numpy.get_include(),
            ],
            name='_cpmorphology2',
            sources=[
                'centrosome/_cpmorphology2.pyx'
            ],
        ),
        Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                'centrosome/src',
                numpy.get_include(),
            ], name='_watershed',
            sources=[
                'centrosome/_watershed.pyx',
                'centrosome/heap_watershed.pxi',
            ],
        ),
        Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                'centrosome/src',
                numpy.get_include(),
            ], name='_propagate',
            sources=[
                'centrosome/_propagate.pyx',
                'centrosome/heap.pxi',
            ],
        ),
        Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                'centrosome/src',
                numpy.get_include(),
            ], name='_filter',
            sources=[
                'centrosome/_filter.pyx'
            ],
        ),
        Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                'centrosome/src',
                numpy.get_include(),
            ], name='_lapjv',
            sources=[
                'centrosome/_lapjv.pyx'
            ],
        ),
        Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                'centrosome/src',
                numpy.get_include(),
            ], name='_convex_hull',
            sources=[
                'centrosome/_convex_hull.pyx'
            ],
        ),
        Extension(
            depends=[
                'centrosome/include/fastemd_hat.hpp',
                'centrosome/include/npy_helpers.hpp'
            ] + glob.glob('centrosome/include/FastEMD/*.hpp'),
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                'centrosome/include',
                'centrosome/include/FastEMD',
                numpy.get_include(),
            ],
            language='c++',
            name='_fastemd',
            sources=[
                'centrosome/_fastemd.pyx'
            ],
        )
    ]
)
