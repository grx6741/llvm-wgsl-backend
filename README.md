# Intro

blah blah blah

> [!WARNING]
> Only Tested on WSL/Ubuntu 20.04 and Debian 10 Bookworm and llvm 19

# Building

Install `llvm-19`

```
sudo apt install llvm-19
```

Clone the repo with all the submodules

```
git clone --recursive git@github.com:grx6741/llvm-wgsl-backend.git
cd dawn
python3 tools/fetch_dawn_dependencies.py --use-test-deps
cd ..
```

Run the Makefile

```
make
```

# NOTES

- ![NVPTX](https://llvm.org/docs/NVPTXUsage.html)
- ![LLVM Lang Ref](https://llvm.org/docs/LangRef.html)
