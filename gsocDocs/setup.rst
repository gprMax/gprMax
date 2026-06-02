GSoC 2026 — Project Setup
=========================

This folder (``gsocDocs/``) holds documentation written alongside the GSoC unit-testing
project. Every pull request gets a corresponding ``.rst`` doc summarising what it does and
why, placed in one of two subfolders:

- ``gsocDocs/feats/`` — one doc per ``feat/*`` PR (new functionality).
- ``gsocDocs/fixes/`` — one doc per ``fix/*`` PR (bug fixes).

The doc should be committed in the same PR it describes.

Branch Structure
----------------

``gsoc26-unit-testing`` is the GSoC integration branch. It is essentially ``devel`` with the
GSoC project layered on top. To avoid a painful merge at the end, we periodically fetch
changes from ``devel`` into this branch and resolve any conflicts incrementally::


At the end of the summer, ``gsoc26-unit-testing`` is squashed into a single commit and
merged into ``devel``.

PR Workflow
-----------

1. Cut a short-lived feature branch off ``gsoc26-unit-testing``::

       git checkout gsoc26-unit-testing
       git checkout -b feat/<short-name>

2. Do the work, commit, push, open a PR **targeting** ``gsoc26-unit-testing`` (not ``devel``).
3. After review and merge, delete the feature branch.

Keeping PRs small and focused makes review easier and reduces merge-conflict surface area.

Notes vs Docs
-------------

- ``notes/`` — gitignored personal scratch space (proposal, todos, experiments).
- ``gsocDocs/`` — committed project documentation intended for mentors and future contributors.