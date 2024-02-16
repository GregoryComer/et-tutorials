# Run a simple model

## Environment eetup

If you haven't already, run the environment setup steps in the [top-level README.md file](../README.md).

## Building

Configure the cmake build system:
```
# All remaining instructions in this README assume that
# you are in this tutorial's directory.
cd 01-run-simple-model

(rm -rf cmake-out \
  && mkdir cmake-out \
  && cd cmake-out \
  && cmake -DBUCK2=buck2 ..)
```

TODO: note that this assumes that buck2 is on your path

Build the example:
```
cmake --build cmake-out --target etsample -j9
```

Run the example:
```
./cmake-out/etsample
```

TODO: this currently fails because it can't find the model. For this first
step, let's just check in the .pte file. A later tutorial can cover the creation
of .pte files.
