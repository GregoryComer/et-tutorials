## Environment setup

1. Clone the repo:
```
git clone git@github.com:dbort/et-tutorials.git
cd et-tutorials
git checkout gasoonjia
```

2. Auto sync executoch repo and switch to main branch

```
git submodule sync && git submodule update --init
cd executorch
git checkout origin/main
```

3. Build ExecuTorch

Please follow the  “Setup Your Environment” step in [ExecuTorch setup page](https://pytorch.org/executorch/main/getting-started-setup.html).
For the last command, please do
```
./install_requirements.sh --pybind
```
instead to build ET with pybinding.

## Export nanoGPT

 Move to nanoGPT directory:
 ```
 cd /path/to/et-tutorials/nanogpt
 ```

 Then export the nanoGPT.pte by

 ```
 python export_nanogpt.py
 ```


## Execute nanoGPT in the runtime
1. Build runtime Environment
```
(rm -rf cmake-out \
  && mkdir cmake-out \
  && cd cmake-out \
  && cmake ..)
```
2. Build nanogpt_runner.cpp
```
cmake --build cmake-out --target nanogpt_runner -j9
```

3. Download vocab json
```
wget https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
```
4. Execute the runner by
```
./cmake-out/nanogpt_runner
```
