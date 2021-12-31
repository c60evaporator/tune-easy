# Author: Kenta Nakamura <c60evaporator@gmail.com>
# Copyright (c) 2020-2021 Kenta Nakamura
# License: BSD 3 clause

from setuptools import setup
import tune_easy

DESCRIPTION = "muscle-tuning: A hyperparameter tuning tool, easy to use even if your brain is made of muscle"
NAME = 'muscle-tuning'
AUTHOR = 'Kenta Nakamura'
AUTHOR_EMAIL = 'c60evaporator@gmail.com'
URL = 'https://github.com/c60evaporator/muscle-tuning'
LICENSE = 'BSD 3-Clause'
DOWNLOAD_URL = 'https://github.com/c60evaporator/muscle-tuning'
VERSION = tune_easy.__version__
PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    'scikit-learn>=0.24.2',
    'matplotlib>=3.3.4',
    'seaborn>=0.11.0',
    'numpy >=1.20.3',
    'pandas>=1.2.4',
    'matplotlib>=3.3.4',
    'optuna>=2.7.0',
    'bayesian-optimization>=1.2.0',
    'mlflow>=1.17.0',
    'lightgbm>=3.2.1',
    'xgboost>=1.4.2',
    'seaborn-analyzer>=0.1.6'
]

EXTRAS_REQUIRE = {
}

PACKAGES = [
    'muscle_tuning'
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Multimedia :: Graphics',
    'Framework :: Matplotlib',
]

with open('README.rst', 'r') as fp:
    readme = fp.read()
with open('CONTACT.txt', 'r') as fp:
    contacts = fp.read()
long_description = readme + '\n\n' + contacts

setup(name=NAME,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=AUTHOR,
      maintainer_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      long_description=long_description,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      packages=PACKAGES,
      classifiers=CLASSIFIERS
    )