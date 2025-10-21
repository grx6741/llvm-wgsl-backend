all: test

dll = ./build/libllvm2wgsl.so

# Build the shared library
$(dll):
	cmake --build build

# Test configuration
test_name = axpy
test_dir = tests
test_device_src = $(test_dir)/$(test_name)/$(test_name).cu
test_device_bin = $(test_dir)/$(test_name)/$(test_name)_device.ll
test_host_src = $(test_dir)/$(test_name)/main.cu
test_host_bin = $(test_dir)/$(test_name)/$(test_name)_host.ll
test_cxx_flags = -Os -S -emit-llvm -fno-discard-value-names -Wno-unknown-cuda-version

# Compile CUDA device code to NVPTX IR
$(test_device_bin): $(test_device_src)
	clang++ --cuda-gpu-arch=sm_70 \
		--cuda-device-only \
		$(test_cxx_flags) \
		-nocudalib \
		$(test_device_src) \
		-o $(test_device_bin)

# Compile CUDA host code to LLVM IR
$(test_host_bin): $(test_host_src)
	clang++ --cuda-gpu-arch=sm_70 \
		--cuda-host-only \
		$(test_cxx_flags) \
		$(test_host_src) \
		-o $(test_host_bin) \
		-I/usr/local/cuda/include

# Run the test
test: $(test_device_bin) $(dll)
	opt-19 -load-pass-plugin $(dll) \
		-passes=llvm2wgsl \
		-disable-output \
		$(test_device_bin)

# Clean up generated files
clean:
	cmake --build build --target clean
	rm -f $(test_device_bin) $(test_host_bin)

# Phony targets
.PHONY: all test clean $(dll)
