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

# Benchmarks

Simple `y = a * x + y` where `x` and `y` are vectors and `a` is a scalar.

## Cuda Device Code

```cpp
__global__ void axpy_kernel( int n, float a, float* x, float* y )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) {
        y[i] = a * x[i] + y[i];
    }
}
```

## Generated WGSL Code

```rust
struct globals_t {
  local_id : vec3<u32>,
  workgroup_id : vec3<u32>,
  num_workgroups : vec3<u32>,
  global_id : vec3<u32>,
  workgroup_size : vec3<u32>,
}

var<private> globals : globals_t;

@group(0u) @binding(0u) var<storage, read> x : array<f32>;

@group(0u) @binding(1u) var<storage, read_write> y : array<f32>;

struct axpy_kernel_params_t {
  a : f32,
  n : i32,
}

@group(0u) @binding(2u) var<uniform> axpy_kernel_params : axpy_kernel_params_t;

fn get_workgroup_id_x() -> i32 {
  return i32(globals.workgroup_id.x);
}

fn get_workgroup_id_y() -> i32 {
  return i32(globals.workgroup_id.y);
}

fn get_workgroup_id_z() -> i32 {
  return i32(globals.workgroup_id.z);
}

fn get_local_id_x() -> i32 {
  return i32(globals.local_id.x);
}

fn get_local_id_y() -> i32 {
  return i32(globals.local_id.y);
}

fn get_local_id_z() -> i32 {
  return i32(globals.local_id.z);
}

fn get_num_workgroups_x() -> i32 {
  return i32(globals.num_workgroups.x);
}

fn get_num_workgroups_y() -> i32 {
  return i32(globals.num_workgroups.y);
}

fn get_num_workgroups_z() -> i32 {
  return i32(globals.num_workgroups.z);
}

fn get_workgroup_size_x() -> i32 {
  return i32(globals.workgroup_size.x);
}

fn get_workgroups_size_y() -> i32 {
  return i32(globals.workgroup_size.y);
}

fn get_workgroups_size_z() -> i32 {
  return i32(globals.workgroup_size.z);
}

fn axpy_kernel() {
  let v = axpy_kernel_params.n;
  let v_1 = axpy_kernel_params.a;
  let v_2 = ((get_workgroup_id_x() * get_workgroup_size_x()) + get_local_id_x());
  if ((v_2 < v)) {
    let v_3 = &(y[v_2]);
    *(v_3) = ((x[v_2] * v_1) + *(v_3));
    return;
  }
}

@compute @workgroup_size(256i, 1i, 1i)
fn wgsl_main(@builtin(local_invocation_id) local_id : vec3<u32>, @builtin(workgroup_id) local_id_1 : vec3<u32>, @builtin(num_workgroups) local_id_2 : vec3<u32>, @builtin(global_invocation_id) global_id : vec3<u32>) {
  globals.local_id = local_id;
  globals.workgroup_id = local_id_1;
  globals.num_workgroups = local_id_2;
  globals.global_id = global_id;
  globals.workgroup_size = vec3<u32>(256u, 1u, 1u);
  axpy_kernel();
}
```

Ran on RTX 3050, 6GB VRAM

| Metric                                | CUDA                | WGSL                      | Notes                                                                 |
| ------------------------------------- | ------------------- | ------------------------- | --------------------------------------------------------------------- |
| **Host → Device Transfer**            | 13.80 ms            | 8.14 ms (Buffer Creation) | WGSL time is just buffer creation; actual host → GPU copy may overlap |
| **Kernel / GPU Execution**            | 1.13 ms             | 32.78 ms                  | WGSL shader execution is ~30× slower than CUDA kernel                 |
| **Device → Host Transfer / Readback** | 6.62 ms             | 8.09 ms                   | Comparable, slightly slower for WGSL                                  |
| **Total Time**                        | 21.56 ms            | 49.09 ms                  | WGSL ~2.3× slower overall                                             |
| **Array Size**                        | 10,000,000 elements | 10,000,000 elements       | Same                                                                  |
| **Data Size**                         | 114.44 MB           | 114.44 MB                 | Same                                                                  |
| **Block / Workgroup Size**            | 256 threads/block   | 256 threads/workgroup     | Same logical partitioning                                             |
| **Number of Blocks / Workgroups**     | 39,063              | 39,063                    | Same                                                                  |
| **Total Threads**                     | 10,000,128          | 10,000,000                | Slight rounding difference                                            |
| **Bandwidth**                         | 98.50 GB/s          | 3.49 GB/s                 | WGSL bandwidth much lower                                             |
| **Performance**                       | 17.63 GFLOPS        | 0.61 GFLOPS               | WGSL performance ~30× lower                                           |
| **Total Time**                        | 22.18 ms            | 49.09 ms                  | WGSL performance ~2× lower                                           |

# NOTES

- ![NVPTX](https://llvm.org/docs/NVPTXUsage.html)
- ![LLVM Lang Ref](https://llvm.org/docs/LangRef.html)
