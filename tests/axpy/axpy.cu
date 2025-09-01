__device__ float get_axpy( float a, float x, float y )
{
    return a * x + y;
}

__global__ void axpy_kernel( int n, float a, float* x, float* y )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) {
        y[i] = get_axpy( a, x[i], y[i] );
    }
}
