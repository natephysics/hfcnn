.. highlight:: shell

============
Installation
============


Stable release
--------------

To install hfcnn, run this command in your terminal:

.. code-block:: console

    $ pip install hfcnn

This is the preferred method to install hfcnn, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Data version control
--------------------

This project uses dvc with the webdav backend.
Please ask the owner to share the config.local file under the .dvc folder with you.

From sources
------------

The sources for hfcnn can be downloaded from the `remote repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/natephysics/hfcnn

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/natephysics/hfcnn/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _remote repo: https://github.com/natephysics/hfcnn
.. _tarball: https://github.com/natephysics/hfcnn/tarball/master
