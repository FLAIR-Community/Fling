# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module setuptools script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from importlib import import_module
from setuptools import setup, find_packages

meta_module = import_module('fling')
meta = meta_module.__dict__
here = os.path.abspath(os.path.dirname(__file__))
with open('README.md', mode='r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name=meta['__TITLE__'],
    version=meta['__VERSION__'],
    description=meta['__DESCRIPTION__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    author=meta['__AUTHOR__'],
    author_email=meta['__AUTHOR_EMAIL__'],
    url='https://github.com/FLAIR-Community/Fling',
    license='Apache License, Version 2.0',
    keywords='Federated Learning Framework',
    packages=[
        *find_packages(include=('fling', 'fling.*')),
        *find_packages(include=('flzoo', 'flzoo.*')),
    ],
    package_data={
        package_name: ['*.yaml', '*.xml', '*cfg', '*SC2Map']
        for package_name in find_packages(include='fling.*')
    },
    python_requires=">=3.7",
    install_requires=[
        'setuptools<=66.1.1', 'yapf==0.29.0', 'torch>=1.7.0', 'torchvision', 'numpy>=1.18.0', 'easydict==1.9',
        'tensorboard>=2.10.1', 'tqdm', 'timm', 'click', 'prettytable', 'einops', 'scipy', 'portalocker', 'six', 'lmdb',
        'imageio[pyav]', 'matplotlib'
    ],
    extras_require={
        'test': [
            'coverage>=5,<=7.0.1',
            'mock>=4.0.3',
            'pytest~=7.0.1',  # required by gym>=0.25.0
            'pytest-cov~=3.0.0',
            'pytest-mock~=3.6.1',
            'pytest-xdist>=1.34.0',
            'pytest-rerunfailures~=10.2',
            'pytest-timeout~=2.0.2'
        ],
        'style': [
            'yapf==0.29.0',
            'flake8<=3.9.2',
            'importlib-metadata<5.0.0',  # compatibility
        ]
    },
    entry_points={'console_scripts': ['fling=fling.cli:cli']},
    classifiers=[
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
