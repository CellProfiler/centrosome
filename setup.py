import Cython.Build
import pkg_resources
import setuptools
import setuptools.command.build_ext
import setuptools.command.test
import sys


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


setuptools.setup(
    author="Allen Goodman",
    author_email="agoodman@broadinstitute.org",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: C",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 2",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering",
    ],
    cmdclass={
        "build_ext": BuildExtension,
        "test": Test
    },
    description="An open source image processing library",
    ext_modules=Cython.Build.cythonize([
        setuptools.Extension(
            name="_cpmorphology",
            sources=[
                "centrosome/src/_cpmorphology.c",
            ]
        ),
        setuptools.Extension(
            name="_propagate",
            sources=[
                "centrosome/_propagate.pyx",
            ],
        ),
        setuptools.Extension(
            name="*",
            language="c++",
            sources=[
                "centrosome/*.pyx",
            ],
        ),
    ]),
    install_requires=[
        "matplotlib",
        "numpy",
        "pillow",
        "scikit-image",
        "scipy",
    ],
    keywords="",
    license="BSD",
    long_description="",
    name="centrosome",
    packages=[
        "centrosome"
    ],
    setup_requires=[
        "cython",
        "numpy",
        "pytest",
    ],
    tests_require=[
        "pytest",
    ],
    url="https://github.com/CellProfiler/centrosome",
    version="1.0.8",
)
