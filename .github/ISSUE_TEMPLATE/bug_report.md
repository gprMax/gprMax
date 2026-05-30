---
name: Bug report
about: Create a report to help us fix a bug in gprMax
title: "[BUG] "
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**Repository version (git commit or pip package)**
Provide the output of `git rev-parse --short HEAD` or the installed package version.

**To Reproduce**
Steps to reproduce the behavior:
1. Provide the model/input file used (attach the .in or a minimal reproduction).
2. List the exact command used to run gprMax.
3. Describe any non-default settings or parameters.

Include a minimal working example or a small input that reproduces the issue.

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened, error messages, stack traces, or incorrect outputs.

**Environment (please complete the following information):**
- OS: (e.g. macOS 12.6, Ubuntu 22.04)
- Python version: (e.g. 3.10.12)
- gprMax version/commit:
- Installed from: (pip / conda / source)
- Any active GPU or special hardware details

**Attachments**
- Include relevant log snippets, the .in model file, small HDF5 outputs if possible, and a link to a minimal example.
- If files are large, provide a link or paste the relevant excerpts.

**Additional context**
Add any other context about the problem here (related issues, recent changes, workarounds tried).


**Checklist**
- [ ] I have searched the existing issues for duplicates.
- [ ] I can reproduce this issue with the latest commit on the default branch.
- [ ] I attached a minimal test case / input file.
