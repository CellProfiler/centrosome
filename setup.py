import glob
import pkg_resources
import setuptools
import setuptools.command.build_ext
import setuptools.command.test
import sys


class Build(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        numpy_includes = pkg_resources.resource_filename("numpy", "core/include")

        for extension in self.extensions:
            if hasattr(extension, "include_dirs") and numpy_includes not in extension.include_dirs:
                extension.include_dirs.append(numpy_includes)

        setuptools.command.build_ext.build_ext.build_extensions(self)


class Test(setuptools.command.test.test):
    user_options = [
        ("pytest-args=", "a", "Arguments to pass to py.test")
    ]

    def initialize_options(self):
        setuptools.command.test.test.initialize_options(self)

        self.pytest_args = []

    def finalize_options(self):
        setuptools.command.test.test.finalize_options(self)
        self.test_args = []

        self.test_suite = True

    def run_tests(self):
        import pytest

        errno = pytest.main(self.pytest_args)

        sys.exit(errno)


if sys.platform.startswith("win"):
    extra_compile_args = None

    extra_link_args = ["/MANIFEST"]
else:
    extra_compile_args = ["-O3"]

    extra_link_args = None

setuptools.setup(
    name="centrosome",
    version="1.0.0",
    description="",
    long_description="",
    url="https://github.com/CellProfiler/centrosome",
    author="Allen Goodman",
    author_email="agoodman@broadinstitute.org",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
    ],
    keywords="",
    packages=[
        "centrosome"
    ],
    install_requires=[
        "scipy",
    ],
    extras_require={
        "development": [

        ],
        "test": [

        ]
    },
    setup_requires=[
        "cython",
        "numpy",
    ],
    tests_require=[
        "pytest"
    ],
    cmdclass={
        "build_ext": Build,
        "test": Test
    },
    ext_modules=[
        setuptools.Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                "centrosome/src",
            ],
            name="_cpmorphology",
            sources=[
                "centrosome/src/cpmorphology.c"
            ],
        ),
        setuptools.Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            name="_cpmorphology2",
            sources=[
                "centrosome/_cpmorphology2.pyx"
            ],
        ),
        setuptools.Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                "centrosome/src",
            ],
            name="_watershed",
            sources=[
                "centrosome/_watershed.pyx",
                "centrosome/heap_watershed.pxi",
            ],
        ),
        setuptools.Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                "centrosome/src",
            ],
            name="_propagate",
            sources=[
                "centrosome/_propagate.pyx",
                "centrosome/heap.pxi",
            ],
        ),
        setuptools.Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                "centrosome/src",
            ],
            name="_filter",
            sources=[
                "centrosome/_filter.pyx"
            ],
        ),
        setuptools.Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                "centrosome/src",
            ],
            name="_lapjv",
            sources=[
                "centrosome/_lapjv.pyx"
            ],
        ),
        setuptools.Extension(
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                "centrosome/src",
            ],
            name="_convex_hull",
            sources=[
                "centrosome/_convex_hull.pyx"
            ],
        ),
        setuptools.Extension(
            depends=[
                "centrosome/include/fastemd_hat.hpp",
                "centrosome/include/npy_helpers.hpp"
            ] + glob.glob("centrosome/include/FastEMD/*.hpp"),
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[
                "centrosome/include",
                "centrosome/include/FastEMD",
            ],
            language="c++",
            name="_fastemd",
            sources=[
                "centrosome/_fastemd.pyx"
            ],
        )
    ]
)
