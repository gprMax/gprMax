# Contributing to gprMax

We welcome contributions!

## CI/CD Pipeline
We use GitHub Actions for our CI/CD pipeline which runs on Ubuntu, macOS, and Windows.
Our pipeline includes:
- Testing across CPU, MPI, and CUDA backends.
- Pre-commit hooks to ensure code format standardisation (Ruff, Codespell, Mypy).
- Automated benchmarking for environmental simulation suites, generating interactive Plotly dashboards and PR comments.
- Automated wheel building and GitHub Releases for tagged versions using `cibuildwheel`.

## Setup your environment
Run `pip install pre-commit && pre-commit install` to set up the hooks.
To build gprMax for your specific backend, use the new `build.py` script:
`python build.py --backend {cpu,mpi,cuda}`

See [docs.gprmax.com](http://docs.gprmax.com) for more details.
