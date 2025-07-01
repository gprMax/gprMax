**********************
Contributing to gprMax
**********************

Thank you for your interest in contributing to gprMax, we really appreciate your time and effort!

If you’re unsure where to start or how your skills fit in, reach out! You can ask us here on GitHub, by leaving a comment on a relevant issue that is already open.

Small improvements or fixes are always appreciated.

If you are new to contributing to `open source <https://opensource.guide/how-to-contribute/>`_, this guide helps explain why, what, and how to get involved.

How can you help us?
--------------------

* Report a bug
* Improve our `documentation <https://docs.gprmax.com/en/devel/>`_
* Submit a bug fix
* Propose new features
* Discuss the code implementation
* Test our latest version which is available through the `devel branch <https://github.com/gprmax/gprMax/tree/devel>`_ on our repository

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

6. Always take a pull from the upstream repository to your devel branch to keep it at par with the main project (updated repository).

.. code-block:: console

    $ git pull upstream devel

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

    Be sure to merge the latest from "upstream" before making a pull request!

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
