#!/bin/bash

set -e -x

yum install -y atlas-devel

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    ${PYBIN}/pip install -e .

    ${PYBIN}/pip wheel /io/ -w wheelhouse/
done

for whl in wheelhouse/*.whl; do
    auditwheel repair $whl -w /io/wheelhouse/
done

for PYBIN in /opt/python/*/bin/; do
    ${PYBIN}/pip install centrosome --no-index -f /io/wheelhouse

    (cd $HOME; ${PYBIN}/tox centrosome)
done
