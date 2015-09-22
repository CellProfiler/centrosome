import pkg_resources
import setuptools
import setuptools.command.build_ext
import setuptools.command.test
import sys

TEST = True

try:
    import Cython.Build
except ImportError:
    TEST = False

EXTENSION = "pyx" if TEST else "c"

EXTENSIONS = [
    setuptools.Extension(
        name="_cpmorphology",
        sources=[
            "centrosome/src/_cpmorphology.c",
        ]
    ),
    setuptools.Extension(
        name="_propagate",
        sources=[
            "centrosome/_propagate." + EXTENSION,
        ],
    ),
    setuptools.Extension(
        name="*",
        language="c++",
        sources=[
            "centrosome/*." + EXTENSION,
        ],
    ),
]

if TEST:
    import Cython.Build

    EXTENSIONS = Cython.Build.cythonize(EXTENSIONS)


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
        "numpy",
    ],
    tests_require=[
        "pytest",
    ],
    cmdclass={
        "build_ext": BuildExtension,
        "test": Test
    },
    ext_modules=EXTENSIONS,
)
