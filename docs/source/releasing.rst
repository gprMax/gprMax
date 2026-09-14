*****************************
Publishing a gprMax release
*****************************

This page is for release maintainers. Users installing gprMax should follow
:ref:`installation`. Release wheels are compiled and tested on GitHub's
Linux, Windows and macOS runners; maintainers do not need to compile them on
their own machines.

The ``Release distributions`` workflow is defined in
``.github/workflows/release.yml``. It reuses the ``Binary distributions``
builders rather than maintaining a second platform matrix. A successful run
produces 12 wheels (Python 3.11--3.13 on four platforms) and one source archive.
See :doc:`testing` for the installed-package checks.

Nothing is published just by pushing a branch, opening a pull request,
pushing a tag or creating a GitHub release. Publication is an explicit manual
workflow action, with a separate approval before the PyPI upload.

Choose one independent destination:

* ``dry-run`` builds, validates and tests the release artifacts without
  uploading them. No PyPI or TestPyPI account setup is needed for this check.
* ``pypi`` performs the same checks, then requests approval to publish to
  PyPI only. TestPyPI access is **not required**.
* ``testpypi`` optionally rehearses an upload and installation using TestPyPI.
  It never publishes to PyPI and is not a prerequisite for a ``pypi`` run.

One-time account setup
======================

GitHub environments
-------------------

In the repository's **Settings → Environments**, configure:

* ``pypi``: selected deployment tags matching ``v*``; no deployment
  branches, plus at least one trusted release maintainer as a **required
  reviewer**. This is what pauses the final upload for approval; the workflow
  YAML alone cannot enforce reviewer settings.
* ``testpypi`` (optional): the same tag-only restriction, if you want to use
  the test index. A required reviewer is optional for this environment.

The release maintainers are Antonis Giannopoulos and Craig Warren. Both are
configured as reviewers for the PyPI environment; approval from either
maintainer is sufficient to release the already-tested files.
Self-review remains allowed, so either maintainer can trigger a run and then
approve its publication after inspecting the results. To require a reviewer
other than the person who triggered the run, enable prevention of self-review.
Review administrator bypass settings too; do not bypass the release checks.

Protect release tags against unauthorised creation, movement and deletion
using the repository's tag rulesets. The deployment tag pattern limits which
refs can publish, but does not itself protect who can create those tags.

PyPI and TestPyPI trusted publishers
---------------------------------------

Log in to PyPI with the account that owns the ``gprMax`` project. Under
**Your projects → gprMax → Manage → Publishing**, add a GitHub trusted
publisher with these exact values:

.. list-table:: Publisher configuration
   :header-rows: 1
   :widths: 35 32 33

   * - Field
     - PyPI
     - TestPyPI (optional)
   * - Project name
     - ``gprMax``
     - ``gprMax``
   * - GitHub owner
     - ``gprMax``
     - ``gprMax``
   * - Repository
     - ``gprMax``
     - ``gprMax``
   * - Workflow filename
     - ``release.yml``
     - ``release.yml``
   * - Environment
     - ``pypi``
     - ``testpypi``

Use only the filename ``release.yml``, not its directory or display name.
Only the PyPI column is needed for production publication. If you later use
the optional TestPyPI route, repeat this registration there: TestPyPI has
separate accounts and publisher settings. The respective project owner must
also arrange project access for both release maintainers on each index being
used; GitHub reviewer permissions do not grant PyPI or TestPyPI project access.
An organisation account is optional; access to the package project is what
matters.

If the organisation has no projects but the new-publisher form says
"This project already exists", ownership of the organisation is not enough.
The name may belong to an existing package project with no uploaded releases.
Obtain access to that project from its current owner, then configure its
**Manage → Publishing** page. Do not delete the project or change the package
name to work around an ownership problem. A project can be transferred into
the organisation once the necessary permissions are in place.

If the project does not exist on an index, use that
account's **Publishing → Add a new pending publisher** form and supply the
project name. A pending publisher does not reserve a name; if someone else
owns it, the owner must grant access before you can publish.

The relevant pages are `PyPI publishing settings
<https://pypi.org/manage/account/publishing/>`_ and `TestPyPI publishing
settings <https://test.pypi.org/manage/account/publishing/>`_. See the official
`existing-project instructions
<https://docs.pypi.org/trusted-publishers/adding-a-publisher/>`_ and
`first-publication instructions
<https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/>`_.

Do not add a PyPI password or a long-lived API token to the repository.
Trusted Publishing supplies short-lived credentials to the upload jobs.
Only those jobs receive ``id-token: write`` permission; building, testing and
running package code take place in separate jobs without that permission.

One-time v3 to v4 branch transition
==================================================

The v4 transition is a separate, coordinated maintainer operation. Merging
the preparation PR or running a package build does not perform it. Until
PyPI project access and Trusted Publishing are configured, keep the public
``master`` branch on v3 and merge reviewed v4 preparation work into ``devel``.
Run the pull-request test and wheel-build workflows in the meantime.

Once publishing is ready and the selected v4 revision has passed its checks:

1. Record the exact tips of ``master`` and ``devel``. Review any relevant
   v3-only fixes for inclusion in v4, and finish or retarget outstanding PRs
   before renaming their branches.
2. Preserve the complete old ``master`` by renaming it to ``legacy/v3``.
   Keep every existing v3 release and tag unchanged. The ``v.3.1.7`` tag
   records the published release, not subsequent v3 development commits;
   it is not a substitute for preserving the final branch tip.
3. Rename the tested ``devel`` branch to ``master`` and explicitly select
   it as the repository's default branch. This changes branch names without
   rebasing the histories or merging the v3 implementation into v4.
4. If retaining a separate development branch, create a fresh ``devel`` from
   the new ``master``. Verify branch protection, required checks, workflow
   triggers and documentation hosting settings after the rename. The v4
   release workflow must now be present on the default branch.
5. Run ``Release distributions`` with ``destination=dry-run`` against the
   new ``master``. Only after it passes should maintainers tag and publish
   the reviewed revision using the procedure below.

Describe ``legacy/v3`` as frozen and not actively maintained; retaining it
does not promise further v3 releases. When publishing the GitHub v4 release,
mark it as the latest release. Older releases remain available for users
who need a specific historical version.

Existing v3 clones tracking ``master`` must not treat the new branch as an
ordinary update: the histories have diverged. Recommend a separate v4
checkout and environment, as described in :doc:`migration_v3_v4`. Contributors
with local changes should preserve their branches before updating tracking
references. Raw branch URLs, source-installation commands and CI references
must also be checked; not every consumer follows a GitHub branch rename.
See `GitHub's branch-renaming guidance
<https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/renaming-a-branch>`_.

Prepare the release revision
============================

1. Merge the reviewed code, documentation and packaging changes. Local,
   uncommitted changes are not included in GitHub-built wheels.
2. Check that the normal test workflows pass for the selected revision, and
   complete the release's physics, GPU and MPI validation. The wheel smoke
   tests exercise CPU execution; they do not replace those validations.
3. Set ``gprMax/_version.py`` to the intended public version, for example
   ``4.0.0rc1`` for a rehearsal or ``4.0.0`` for the final release. Review the
   migration guide and release notes for that revision.
4. Ensure both ``release.yml`` and the reusable ``wheels.yml`` exist in the
   release revision. For manual dispatch, ``release.yml`` must also exist on
   the repository's default branch. Do not assume that a workflow present
   only on a development branch will be available in the Actions menu.

The release workflow defaults to ``dry-run``. This mode can run on a branch
and validates the complete artifact set without contacting either upload
endpoint. All wheels pass their platform-specific installed-wheel tests
during the build. An additional job selects the Linux Python 3.12 wheel from
the collected ``release-distributions`` artifact, verifies its SHA-256 against
the manifest, installs it with dependencies from PyPI, and runs the CPU wheel
smoke test outside the checkout. The package itself is selected locally with
``--no-index``; no published gprMax package or TestPyPI access is required.

From GitHub Actions, select **Release distributions → Run
workflow**, choose the intended branch and leave ``destination`` as
``dry-run``. Alternatively, with the GitHub CLI authenticated:

.. code-block:: console

    gh workflow run release.yml --repo gprMax/gprMax --ref master -f destination=dry-run

Here ``master`` means the v4 branch after the transition above. For later
rehearsals, select the branch containing the reviewed release work. The
workflow uses the selected revision's version, without
importing or running gprMax to discover it.

Rehearse or publish
======================

For either index, first create and push an immutable version tag on the
reviewed commit. Following the existing v3 convention, the tag must exactly
match the package version with a ``v.`` prefix: ``4.0.0rc1`` requires
``v.4.0.0rc1``; ``4.0.0`` requires ``v.4.0.0``. Publishing from a branch,
pull-request ref or mismatched tag is rejected, including the alternate
spelling ``v4.0.0``.

The Git tag, GitHub release title and PyPI package version are distinct:

* Git tag: ``v.4.0.0``.
* GitHub release title: ``v.4.0.0 (Caol Ila)``.
* Package version in ``gprMax/_version.py`` and on PyPI: ``4.0.0``.

The codename belongs in the release title, not the tag or package version.
Create the tag before starting the publication workflow; publish the matching
GitHub release page after verifying the PyPI upload, as described below.

The commands below upload real files to the selected indexes. Use a fresh
version for each rehearsal; do not run them merely to validate the YAML.

Optional TestPyPI rehearsal
--------------------------------

Skip this section if TestPyPI access is unavailable. A successful ``dry-run``
is the pre-publication rehearsal for a PyPI-only release. It checks package
building and installation, but cannot verify the PyPI account's Trusted
Publisher registration or an actual index upload.

After preparing and pushing the ``v.4.0.0rc1`` tag:

.. code-block:: console

    gh workflow run release.yml --repo gprMax/gprMax --ref v.4.0.0rc1 -f destination=testpypi

This builds, validates and uploads to TestPyPI only. It downloads the Linux
Python 3.12 wheel from TestPyPI, verifies its SHA-256 against the built
artifact, installs it with dependencies from normal PyPI, and runs the wheel
smoke test outside the checkout. The other wheels have already passed their
platform-specific installed-wheel tests during the build.

Dependencies are not resolved from a mixture of PyPI and TestPyPI: the gprMax
wheel is downloaded with ``--no-deps`` from TestPyPI, then installed with
dependencies coming from PyPI. A short retry window allows the test index to
expose newly uploaded files.

Final publication
-----------------

For the reviewed ``v.4.0.0`` tag, with that version not already uploaded:

.. code-block:: console

    gh workflow run release.yml --repo gprMax/gprMax --ref v.4.0.0 -f destination=pypi

This run builds and checks the release, including the artifact installation
test described above, then pauses at the protected ``pypi`` environment. It
does not upload to TestPyPI, contact its index or depend on its verification
job. Inspect the build/test results, the selected commit and version, and
the ``release-manifest`` artifact. In GitHub Actions, choose
**Review deployments → pypi → Approve and deploy**
only when ready to make the release public.

The final job uploads the same ``release-distributions`` artifact to PyPI;
it does not rebuild it. All platform builds must have succeeded. Missing,
duplicate or unexpected wheels, wrong package versions and metadata/tag
mismatches stop the workflow before an upload. The manifest records the
size and SHA-256 of every file.

After publication, verify an installation from PyPI in a clean environment,
publish or update the matching GitHub release notes and documentation, and
announce the release. GitHub release pages and PyPI releases are separate:
this workflow neither creates GitHub release pages nor attaches assets to
them. GPU and MPI users still need the optional runtimes described in
:doc:`accelerators` and :doc:`hpc`.

Failures and reruns
======================

Do not move a published tag, silently reuse an existing filename or enable
``skip-existing`` to conceal a mismatch. The upload jobs deliberately fail
on duplicates. Each destination uploads to its selected index only: a prior
TestPyPI upload does not cause a duplicate there when you later choose
``pypi``. However, separate workflow runs rebuild their artifacts; the new
run must pass all checks and is not a byte-for-byte promotion of the previous
run. Within each run, the upload uses that run's validated and tested
``release-distributions`` artifact without rebuilding it. Use a fresh
release-candidate version for each optional TestPyPI rehearsal, and do not
repeat an upload of the same files to the same index.

If an upload fails after publishing some files, inspect the index and compare
their hashes with the saved manifest before deciding how to recover. Do not
blindly rerun the whole workflow. Correcting package contents normally needs
a new version and tag. If only a later check fails after a successful upload,
rerunning the failed jobs can reuse the existing artifacts without repeating
the successful upload, subject to GitHub's artifact retention period.

An environment/publisher mismatch should be fixed in the account settings,
not by adding credentials to a build job. Nothing in local testing or a
``dry-run`` proves the external OIDC registration: it is exercised by an
authorised upload to the selected index. A successful TestPyPI upload also
does not verify the separate PyPI registration. See the `PyPA publishing guide
<https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>`_
for the underlying GitHub/PyPI mechanism.
