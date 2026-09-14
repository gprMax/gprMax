**********************
Contributing to gprMax
**********************

Thank you for your interest in contributing to gprMax, we really appreciate your time and effort!

If you’re unsure where to start or how your skills fit in, reach out! You can ask us here on GitHub, by leaving a comment on a relevant issue that is already open.

Small improvements or fixes are always appreciated.

Release maintainers should follow :doc:`releasing` for GitHub wheel builds,
optional TestPyPI rehearsals and approval-gated PyPI publication.

If you are new to contributing to `open source <https://opensource.guide/how-to-contribute/>`_, this guide helps explain why, what, and how to get involved.

Building the documentation
--------------------------

The User Guide can be built from a source checkout without installing or
compiling gprMax. A separate Python 3.12 environment is recommended: using a
full development environment can hide missing documentation dependencies.
From the repository root on Linux or macOS:

.. code-block:: console

    $ python3.12 -m venv .venv-docs
    $ source .venv-docs/bin/activate
    $ python -m pip install -r docs/requirements.txt
    $ make -C docs html

The build treats Sphinx warnings as errors. The generated HTML starts at
``docs/build/index.html``.

On Windows, create the environment with ``py -3.12 -m venv .venv-docs`` and
activate it with ``.venv-docs\Scripts\Activate.ps1`` in PowerShell. After
installing the same requirements, build HTML with
``python -m sphinx -b html -aE -W --keep-going docs/source docs/build``.

The ``Documentation`` GitHub Actions workflow builds HTML and LaTeX sources
using only ``docs/requirements.txt``, matching Read the Docs' Python version.
It does not install the solver or the test dependencies. Both CI and Read
the Docs treat Sphinx warnings as errors, so a missing API import cannot
silently produce incomplete reference pages. A successful CI build provides
a downloadable ``documentation-preview`` artifact; it does not publish the
hosted documentation. That deployment is performed separately by Read the
Docs after a successful build of the selected branch or tag.

A PDF version can be generated from the same sources. A TeX distribution that
provides XeLaTeX and ``latexmk`` is additionally required (for example, TeX
Live on Linux and macOS or MiKTeX on Windows):

.. code-block:: console

    $ make -C docs latexpdf

On native Windows, the equivalent Sphinx make-mode command is:

.. code-block:: console

    > sphinx-build -M latexpdf docs/source docs/build -W --keep-going

The resulting file is ``docs/build/latex/gprMax.pdf``. Read the Docs also
builds a downloadable PDF for each published documentation version.

How can you help us?
--------------------

* Report a bug
* Improve our `documentation <https://docs.gprmax.com/en/latest/>`_
* Submit a bug fix
* Propose new features
* Discuss the code implementation
* Test the current maintained code in the `master branch <https://github.com/gprmax/gprMax/tree/master>`_ of the repository

How to Contribute
-----------------

In general, we follow the "fork-and-pull" Git workflow.

1. Fork the gprMax repository
2. Clone the repository

.. code-block:: console

   $ git clone https://github.com/Your-Username/gprMax.git

3. Navigate to the project directory.

.. code-block:: console

    $ cd gprMax

4.  Add a reference(remote) to the original repository.

.. code-block:: console

    $ git remote add upstream https://github.com/gprMax/gprMax.git

5.  Check the remotes for this repository.

.. code-block:: console

    $ git remote -v

6. Update your local ``master`` branch from the upstream repository before
   creating a feature branch.

.. code-block:: console

    $ git switch master
    $ git pull --ff-only upstream master

7. Create a new branch.

.. code-block:: console

    $ git checkout -b <your_branch_name>

8. Run the following command before you commit your changes to ensure that your code is formatted correctly:

.. code-block:: console

    $ pre-commit run --all-files

9.  Make the changes you want to make and then add

.. code-block:: console

    $ git add .

10.  Commit your changes. To contribute to this project

.. code-block:: console

    $ git commit  -m "<commit subject>"

11.  Push your local branch to your fork

.. code-block:: console

    $ git push -u origin <your_branch_name>

12.  Submit a Pull request so that we can review your changes

.. note::

    Be sure to incorporate the latest ``upstream/master`` before making a pull
    request.

Feature and Bug reports
-----------------------

We use GitHub issues to track bugs and features. Report them by opening a `new issue <https://github.com/gprMax/gprMax/issues>`_.

Code review process
-------------------

The Pull Request reviews are done frequently. Try to explain your PR as much as possible using our template. Also, please make sure you respond to our feedback/questions about the PR.

Community
---------

Please use our `Google Group <https://groups.google.com/g/gprmax>`_ (Forum) for comments, interaction with other users, chat, and general discussion on gprMax, GPR, and FDTD.

Checkout our website `gprmax.com <https://www.gprmax.com/>`_ for more information and updates.
