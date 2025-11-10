# InsPy
Inelastic Neutron Scattering Resolution Calculation Python Package

The latest version InsPy 0.2.6 can be found in the following link:

https://github.com/gcdengansto/inspy-v0.2.6


InsPy
=========

.. warning::
    New releases may not be backwards compatibile.

.. warning::
    Official support for Python 2.7 and Python 3.3 has been discontinued.


InsPy is a python library with commonly used tools for neutron scattering measurements, primarily for Triple Axis Spectrometer data.

    * Triple Axis Spectrometer resolution function calculation (based on ResLib), including
        * Resolution ellipses
        * Convolution Fitting
    * Least-Squares fitting (custom interface for scipy.optimize.leastsq using lmfit features), including
        * Built-in physical models
    * GUI interface for Resolution Calculation and Resolution Convolution Fitting without coding
        * Two GUI interfaces for convolution fitting.



Inspy is a python package to conduct the resolution calculation for the inelastic neutron scattering experiment with triple-axis spectrometers. 
Inspy is developed on the basis of the python package Neutronpy, which was developed by David M Fobes by translating the functions in the MATLAB
 library Reslib3.4c by Andrew Zheludev. 
Inspy involves significant upgrades in comparison to Neutronpy. Inspy is able to do the 3D resolution calculation and plot 3D resolution functions, fit the triple-axis spectrometer by convolting the instrument resolution. The bugs in the resolution calculation have been corrected. Two GUI interfaces were implemented to do the reolsution calcualtion and data fitting. 
InsPy is a work-in-progress and as such, still has many bugs, so use at your own risk; see Disclaimer. To report bugs or suggest features see Contributions.

Requirements
------------
The following packages are required to install this library:

* ``Python >= 3.4``
* ``numpy >= 1.19.0``
* ``scipy >= 1.0.0``
* ``lmfit >= 1.0.2``
* ``matplotlib >= 3.1.0``
* ``h5py``

The following package is required to use the ``inspy`` entry-point gui optional feature

* ``pyqt5 >= 5.4.1``

The following packages are required to test this library:

* ``pytest >= 3``



Installation
------------

Local installation: go to the folder of InsPy

    pip install -e .

See Installation for more detailed instructions.

Documentation
-------------
Please refer to the following article for detailed information about this package InsPy:

[TasVisAn and InsPy: Python packages for triple-axis spectrometer data visualization, analysis, instrument resolution calculation and convolution](https://onlinelibrary.wiley.com/iucr/doi/10.1107/S1600576725008180)

by Guochu Deng* and Garry J. McIntyre, [Journal Of Applied Crystallography](https://journals.iucr.org/j/) Volume 58, Page 1-14,  2025

The DOI of this article is as follows:

[https://doi.org/10.1107/S1600576725008180](https://doi.org/10.1107/S1600576725008180)

Please find [the tutorial](https://github.com/gcdengansto/inspy/blob/main/examples/TasVisAn_Demo.ipynb)  in [the examples folder](https://github.com/gcdengansto/inspy/examples).

The video clips demonstrating how to run the GUIs for data fitting in this package can be found here:
[InsPy_demon](https://doi.org/10.1107/S1600576725008180/te5154sup3.mp4)


Contributions
-------------
Feature requests and bug reports can be made using the GitHub issues interface. 


Copyright & Licensing
---------------------
Copyright (c) 2020-2025, Guochu Deng, Released under terms in MIT LICENSE.

The source code of Inspy is partially from the python package [Neutronpy](https://neutronpy.github.io/), which was developed by D. M Fobes with signifiant changes and updates. The source code for the triple-axis spectrometer resolution calculation was partially based on or translated from the MATLAB library [ResLib 3.4c] (http://www.neutron.ethz.ch/research/resources/reslib),  which was originally developed by A. Zheludev, ETH Zuerich.

If the source code in this Python package is used for data analysis for publications, please cite the article mentioned above. Namely, [TasVisAn and InsPy: Python packages for triple-axis spectrometer data visualization, analysis, instrument resolution calculation and convolution](https://onlinelibrary.wiley.com/iucr/doi/10.1107/S1600576725008180)

Disclaimer
----------
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

