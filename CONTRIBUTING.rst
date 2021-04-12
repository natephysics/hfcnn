.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/natephysics/hfcnn/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

If you want quick feedback, it is helpful to mention speicific developers
(@devloper_name) or @all. This will trigger a mail to the corresponding developer(s).

Fix Bugs
~~~~~~~~

Look through the repository issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the remote issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

`hfcnn` could always use more :ref:`documentation<Documentation>`, whether as part of the
official `hfcnn` docs, in docstrings, or even on the web in blog posts,
articles, and such.

Write Unittests or Doctests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`hfcnn` profits a lot from better :ref:`testing<Testing>`. We encourage you to add unittests 
(in the `tests` directory) or doctests (as part of docstrings or in the documentation).

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an `Issue <https://github.com/natephysics/hfcnn/issues>`_.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `hfcnn` for local development.

1. Fork the `hfcnn` repo.
2. Clone your fork locally::

    $ git clone git@github.com:natephysics/hfcnn.git

3. Set up your fork for local development::

    $ cd hfcnn/
    $ pip install .[dev]

4. Step 3. already installed `pre-commit <https://pre-commit.com/>`_. Initialize it by running::

    $ pre-commit install

5. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

6. When you're done making changes, check that your changes pass flake8 and the
   tests::

    $ make test

7. Commit your changes and push your branch to origin::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the repository website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.5, 3.6, 3.7 and 3.8, and for PyPy. Check
   https://github.com/natephysics/hfcnn/-/merge_requests
   and make sure that the tests pass for all supported Python versions.

Testing
-------

To run tests, use::

    $ make test

To run a subset of tests, you have the following options::

    $ pytest tests/test_package.py

    $ pytest tests/test_package.py::Test_hfcnn::test_version_type

    $ pytest --doctest-modules docs/usage.rst

    $ pytest --doctest-modules hfcnn/core.py -k "MyClass.funciton_with_doctest"

Use the '--trace' option to directly jump into a pdb debugger on fails. Check out the coverage of your api with::

    $ make coverage

Documentation
-------------
To compile the documentation (including automatically generated module api docs), run::

    $ make doc

Use doctests as much as possible in order to have tested examples in your documentation.

Styleguide
-----------
Please follow the `google style guide <https://google.github.io/styleguide/pyguide.html>`_ illustrated
by `this example <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed.
Then run::

    $ bump2version patch # possible: major / minor / patch
    $ git push
    $ git push --tags

or use the convenient alias for the above (patch increases only)::

    $ make publish

The CI will then deploy to PyPI if tests pass.
