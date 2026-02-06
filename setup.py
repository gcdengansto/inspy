#!/usr/bin/env python
"""INSPy: Neutron scattering tools for scientific data analysis in python

INSPy is a collection of commonly used tools aimed at facilitating the
analysis of neutron scattering data. INSPy is built primarily using the
numpy and scipy python libraries, with a translation of ResLib 3.4c (MatLab)
routines for Instrument resolution calculations.

"""

from setuptools import setup, find_packages

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Natural Language :: English
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Topic :: Scientific/Engineering :: Physics
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Operating System :: Unix
Operating System :: MacOS :: MacOS X
"""

DOCLINES = __doc__.split("\n")


def setup_package():
    """Setup package function"""
    
    metadata = dict(
        name='inspy',
        version='0.2.7',
        description=DOCLINES[0],
        long_description="\n".join(DOCLINES[2:]),
        long_description_content_type='text/plain',
        author='Guochu Deng',
        author_email='guochu.deng@ansto.gov.au',
        maintainer='Guochu Deng',
        maintainer_email='guochu.deng@ansto.gov.au',
        download_url='https://github.com/gcdengansto/inspy/releases',
        url='https://github.com/gcdengansto/inspy',
        license='MIT',
        platforms=["Windows", "Linux", "Mac OS X", "Unix"],
        python_requires='>=3.6',
        install_requires=[
            'numpy>=1.15.0',  # Fixed typo: was 1.50
            'scipy>=1.0',
            'pandas',
            'matplotlib>=2.0',
            'lmfit>=1.2.0',
            'h5py',
            'QtPy',
            'plotly>=4.5'
        ],
        setup_requires=[],
        tests_require=[],
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        ext_package='inspy',
        package_data={
            'inspy': [
                'database/*.json',
                'gui/ui/*.ui'  # Fixed: removed leading slash
            ]
        },
        packages=find_packages(exclude=['tests', 'tests.*']),
        entry_points={
            "console_scripts": [
                "inspy=inspy.gui.main_gui:main"
            ]
        },
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()