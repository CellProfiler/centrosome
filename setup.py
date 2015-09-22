import glob
import pkg_resources
import setuptools
import setuptools.command.build_ext
import setuptools.command.test
import sys
import Cython.Build

class BuildExtension(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        numpy_includes = pkg_resources.resource_filename("numpy", "core/include")

        for extension in self.extensions:
            if hasattr(extension, "include_dirs") and numpy_includes not in extension.include_dirs:
                extension.include_dirs.append(numpy_includes)

            extension.include_dirs.append("centrosome/include")

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
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 2",
        "Topic :: Scientific/Engineering",
    ],
    keywords="",
    packages=[
        "centrosome"
    ],
    install_requires=[
        "numpy",
        "scipy",
    ],
    setup_requires=[
        "cython",
        "numpy",
    ],
    tests_require=[
        "pytest"
    ],
    cmdclass={
        "build_ext": BuildExtension,
        "test": Test
    },
    ext_modules=Cython.Build.cythonize([
        setuptools.Extension(
            name="_cpmorphology",
            sources=[
                "centrosome/src/cpmorphology.c"
            ]
        ),
        setuptools.Extension(
            name="_propagate",
            sources=[
                "centrosome/_propagate.pyx",
                "centrosome/heap.pxi"
            ],
        ),
        setuptools.Extension(
            name="_fastemd",
            sources=[
                "centrosome/_fastemd.pyx",
            ],
            depends=[
                "centrosome/include/fastemd_hat.hpp",
                "centrosome/include/npy_helpers.hpp"
            ] + glob.glob("centrosome/include/*.hpp"),
            language="c++"
        ),
        setuptools.Extension(
            name="*",
            sources=[
                "centrosome/*.pyx",
            ],
            include_dirs=[
                "centrosome/include",
            ],
            language="c++",
        ),
    ]),
)
