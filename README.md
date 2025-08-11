# straxion
[![Test package](https://github.com/WashU-Astroparticle-Lab/straxion/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/WashU-Astroparticle-Lab/straxion/actions/workflows/pytest.yml)
[![Coverage Status](https://coveralls.io/repos/github/WashU-Astroparticle-Lab/straxion/badge.svg?branch=main)](https://coveralls.io/github/WashU-Astroparticle-Lab/straxion?branch=main)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![CodeFactor](https://www.codefactor.io/repository/github/washu-astroparticle-lab/straxion/badge)](https://www.codefactor.io/repository/github/washu-astroparticle-lab/straxion)

[`strax`](https://github.com/AxFoundation/strax)-based time series analysis for single photon counting experiments, inspired by [the `straxen` structure]([url](https://github.com/XENONnT/straxen/tree/master)). `straxion` is currently used by (new) [*QAULIPHIDE*](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.231001) dark photon search experiments.

## Installation
```
cd straxion
pip install -e ./ --user
```

## Tutorials
- See [this document](https://strax.readthedocs.io/en/latest/) for `strax` related functionality.
- Check `notebooks/` for `straxion` tutorials in the *QUALIPHIDE* context.

## Tests
The test pipeline in `tests/` is built based on synthesized data contained in the released tag `test-data`. Please refer to `docs/SECURE_TEST_DATA_SETUP.md` for how to update or add test data. The tests are triggered every time a commit is made in a pull request, and one can refer to `Actions` of this repo for examples.

## Contribution
- **This is a public repo. Please avoid putting sensitive information about the experiments in either plots or code.**
- Please make a pull request for any development, except for documentation-related edits.
- Note that [`pre-commit`](https://pre-commit.com) will automatically examine and try to fix the code styles for each commit. Sometimes the fix cannot be done automatically, and you will need to update manually at your discretion.
- Please avoid bumping versions of this package or the plugins without communication with other developers.
