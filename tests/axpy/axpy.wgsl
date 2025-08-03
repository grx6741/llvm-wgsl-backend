// Define a uniform buffer for scalar inputs like 'a' and 'n'.
// This struct will be bound to group 0, binding 0.
struct Params {
    n: u32,
    a: f32,
    // WGSL requires uniform buffer members to be 4-byte aligned.
    // Padding ensures 'a' starts at an offset divisible by 4.
    // In this case, f32 is 4 bytes, u32 is 4 bytes, so no explicit padding is needed
    // if 'n' is also u32. But it's good practice to be mindful of alignment.
    // For instance, if 'n' were u16, you'd need padding.
};
@group(0) @binding(0)
var<uniform> params: Params;

// Define a storage buffer for input array 'x'.
// This will be bound to group 0, binding 1.
// 'read' access is sufficient as 'x' is not modified.
@group(0) @binding(1)
var<storage, read> x: array<f32>;

// Define a storage buffer for input/output array 'y'.
// This will be bound to group 0, binding 2.
// 'read_write' access is needed as 'y' is read from and written to.
@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

// The main entry point for the compute shader.
// `@compute` declares this as a compute shader entry point.
// `@workgroup_size(256)` specifies the local workgroup size.
// This is analogous to CUDA's `blockDim.x`. You can change this value
// based on your desired workgroup size.
// `@builtin(global_invocation_id)` provides the unique global ID for each
// invocation across the entire dispatch, analogous to `blockIdx.x * blockDim.x + threadIdx.x`
// in CUDA.
@compute @workgroup_size(256) // Choose your desired local workgroup size (e.g., 64, 128, 256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Extract the linear global index.
    let i: u32 = global_id.x;

    // Perform bounds checking, similar to `if (i < n)` in CUDA.
    if (i < params.n) {
        y[i] = params.a * x[i] + y[i];
    }
}
