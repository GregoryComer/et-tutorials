## Environment setup

Clone the repo:
```
git clone git@github.com:dbort/et-tutorials.git
cd et-tutorials
```

Set up the submodules. Avoid using `--recursive` because we don't need to pull
in all of the many `pytorch` submodules.
```
git submodule sync && git submodule update --init \
  && (cd executorch && git submodule sync && git submodule update --init)
```

Install `buck2` and ensure that it's on your path:
```
TODO
```

Install conda and create and activate an environment:
```
conda create -yn et-tutorials python=3.10.0
conda activate et-tutorials
```

Install cmake:
```
conda install cmake
```

Install executorch requirements:
```
(cd executorch && ./install_requirements.sh)
```

(TODO: handle flatc)

## Building tutorials

See the README.md files in each tutorial directory for more instructions:

* [Run a simple model](01-run-simple-model/README.md)
