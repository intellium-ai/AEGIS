#!/usr/bin/bash

PYTHON_PATH=$(which python3)

poetry env use $PYTHON_PATH
poetry shell

pip install --upgrade pip==23.0.1
pip install wheel==0.38.4 --upgrade
pip install setuptools==66 --upgrade
pip install --no-use-pep517 gym==0.21.0
poetry install