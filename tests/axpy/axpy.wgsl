struct Params {
    n: u32,
    a: f32,
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> x: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

struct Globals {
    global_id: vec3<u32>,
}

var<private> globals: Globals;

fn axpy(a: f32, x_val: f32, y_val: f32) -> f32 {
    return a * x_val + y_val;
}

fn axpy_kernel() {
    let i: u32 = globals.global_id.x;
    if (i < params.n) {
        y[i] = axpy(params.a, x[i], y[i]);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    globals.global_id = global_id;
    axpy_kernel();
}
