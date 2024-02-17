# Export a simple model

## Environment eetup

If you haven't already, run the environment setup steps in the [top-level
README.md file](../README.md).

TODO: Also need to `(cd executorch && ./install_requirements.sh)`

## Running

Run the script:
```
# All remaining instructions in this README assume that
# you are in this tutorial's directory.
cd 02-export-simple-model

./export_simple_model.py
```

This should create a file named `SimpleModel.pte` in the current directory.
This model is compatible with the `etsample` tool in ../01-run-simple-model.
