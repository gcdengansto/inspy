#!/usr/bin/env python
"""INSPy: Neutron scattering tools for scientific data analysis in python

INSPy is a collection of commonly used tools aimed at facilitating the
analysis of neutron scattering data. INSPy is built primarily using the
numpy and scipy python libraries, with a translation of ResLib 3.4c (MatLab)
routines for Instrument resolution calculations.

"""

import os
import re
import subprocess
import warnings
from math import ceil, log10

from setuptools import setup

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Natural Language :: English
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Scientific/Engineering :: Physics
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Operating System :: Unix
Operating System :: MacOS :: MacOS X
"""

DOCLINES = __doc__.split("\n")



def setup_package():
    r"""Setup package function
    """

    metadata = dict(name='inspy',
                    version='0.2.5',
                    description=DOCLINES[0],
                    long_description="\n".join(DOCLINES[2:]),
                    author='Guochu Deng',
                    author_email='guochu.deng@ansto.gov.author',
                    maintainer='gcdeng',
                    download_url='https://github.com/inspy/inspy/releases',
                    url='https://github.com/inspy/inspy',
                    license='MIT',
                    platforms=["Windows", "Linux", "Mac OS X", "Unix"],
                    install_requires=['numpy>=1.10', 'scipy>=1.0', 'matplotlib>=2.0', 'lmfit>=0.9.5', 'h5py','plotly>=4.5'],
                    setup_requires=[],
                    tests_require=[],
                    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
                    ext_package='inspy',
                    package_data={'inspy': ['database/*.json', 'ui/*.ui']},
                    packages=['inspy', 'inspy.crystal', 'inspy.data', 'inspy.fileio',
                              'inspy.fileio.loaders', 'inspy.instrument', 'inspy.scattering',
                              'inspy.lsfit'],
                    entry_points={"console_scripts": ["inspy=inspy.gui:launch"]}, )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
