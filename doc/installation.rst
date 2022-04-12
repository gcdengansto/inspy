Installing INSPy
====================
The following explains various methods for installing inspy on your system.


Anaconda Cloud - conda -- **Recommended**
-----------------------------------------
The recommended method for installing `inspy <https://anaconda.org/inspy>`_ is using `Anaconda <http://branding-continuum-content.pantheonsite.io/downloads>`_. Once Anaconda is installed, from the command line::

    conda install -c inspy inspy

``pyqt5`` is necessary for the inspy resolution gui executed from the command line with ``inspy``, but optional for all other features of inspy.

Python Package Index - pip
--------------------------
The next method for installing `inspy <https://pypi.python.org/pypi/inspy>`_ is using `pip <https://pip.pypa.io/en/latest/installing.html>`_.

If you do not already have ``pip``, to install it first download `get-pip.py <https://bootstrap.pypa.io/get-pip.py>`_ and run it with the following command::

    python get-pip.py

With ``pip`` installed, you can install the latest version of inspy with the command::

    pip install inspy

To install a specific version of inspy, append ``=={version}`` to the above command, *e.g.*::

    pip install inspy==1.1.0b2

New releases will be pushed to the package index. If you wish to install the development version, you will need to follow the instructions for installation from source.

Installation from Source
------------------------
To install from source, either download the `master branch source from Github <https://github.com/inspy/inspy/archive/master.zip>`_ or clone the repository::

    git clone https://github.com/inspy/inspy.git

From inside of the inspy directory install the package using::

    python setup.py install

If you want to install the development version, you can either download the `development branch source from Github <https://github.com/inspy/inspy/archive/develop.zip>`_, as above, after switch to the ``develop`` branch, or clone the repository and checkout the branch and install::

    git clone https://github.com/inspy/inspy.git
    git fetch
    git checkout develop
    python setup.py install

