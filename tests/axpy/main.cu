#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK( call )                                                                         \
    do {                                                                                           \
        cudaError_t error = call;                                                                  \
        if ( error != cudaSuccess ) {                                                              \
            fprintf( stderr,                                                                       \
                     "CUDA error at %s:%d: %s\n",                                                  \
                     __FILE__,                                                                     \
                     __LINE__,                                                                     \
                     cudaGetErrorString( error ) );                                                \
            exit( EXIT_FAILURE );                                                                  \
        }                                                                                          \
    } while ( 0 )

// AXPY kernel
__global__ void axpy_kernel( int n, float a, const float* x, float* y )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) {
        y[i] = a * x[i] + y[i];
    }
}

// Timing structure
typedef struct
{
    float memcpy_host_to_device;
    float kernel_execution;
    float memcpy_device_to_host;
    float total;
} Timings;

// Run AXPY with detailed timing
Timings run_axpy_cuda( int n, float a, const float* h_x, float* h_y, float* h_result )
{
    Timings timings{};
    cudaEvent_t start, stop;
    float milliseconds;

    // Create CUDA events for timing
    CUDA_CHECK( cudaEventCreate( &start ) );
    CUDA_CHECK( cudaEventCreate( &stop ) );

    // Allocate device memory
    float *d_x, *d_y;
    size_t bytes = n * sizeof( float );
    CUDA_CHECK( cudaMalloc( &d_x, bytes ) );
    CUDA_CHECK( cudaMalloc( &d_y, bytes ) );

    // Time: Host to Device transfer
    CUDA_CHECK( cudaEventRecord( start ) );
    CUDA_CHECK( cudaMemcpy( d_x, h_x, bytes, cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( d_y, h_y, bytes, cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaEventRecord( stop ) );
    CUDA_CHECK( cudaEventSynchronize( stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &milliseconds, start, stop ) );
    timings.memcpy_host_to_device = milliseconds;

    // Time: Kernel execution
    int blockSize = 256;
    int numBlocks = ( n + blockSize - 1 ) / blockSize;

    CUDA_CHECK( cudaEventRecord( start ) );
    axpy_kernel< < < numBlocks, blockSize > > >( n, a, d_x, d_y );
    CUDA_CHECK( cudaEventRecord( stop ) );
    CUDA_CHECK( cudaEventSynchronize( stop ) );
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaEventElapsedTime( &milliseconds, start, stop ) );
    timings.kernel_execution = milliseconds;

    // Time: Device to Host transfer
    CUDA_CHECK( cudaEventRecord( start ) );
    CUDA_CHECK( cudaMemcpy( h_result, d_y, bytes, cudaMemcpyDeviceToHost ) );
    CUDA_CHECK( cudaEventRecord( stop ) );
    CUDA_CHECK( cudaEventSynchronize( stop ) );
    CUDA_CHECK( cudaEventElapsedTime( &milliseconds, start, stop ) );
    timings.memcpy_device_to_host = milliseconds;

    // Calculate total time
    timings.total =
        timings.memcpy_host_to_device + timings.kernel_execution + timings.memcpy_device_to_host;

    // Cleanup
    CUDA_CHECK( cudaFree( d_x ) );
    CUDA_CHECK( cudaFree( d_y ) );
    CUDA_CHECK( cudaEventDestroy( start ) );
    CUDA_CHECK( cudaEventDestroy( stop ) );

    return timings;
}

// Validate results
void validate_results(
    const float* x, const float* y_original, const float* y_result, float a, int n )
{
    printf( "\nValidation:\n" );
    printf( "Checking first 10 elements:\n" );
    printf( "Expected: y[i] = a*x[i] + y_original[i] = %.2f*x[i] + y_original[i]\n\n", a );

    int errors = 0;
    float max_error = 0.0f;

    for ( int i = 0; i < ( n < 10 ? n : 10 ); i++ ) {
        float expected = a * x[i] + y_original[i];
        float actual = y_result[i];
        float error = fabsf( expected - actual );

        if ( error > max_error ) {
            max_error = error;
        }

        if ( error > 1e-5f ) {
            errors++;
        }

        printf( "[%d] x=%.2f, y_orig=%.2f => expected=%.2f, got=%.2f %s\n",
                i,
                x[i],
                y_original[i],
                expected,
                actual,
                ( error > 1e-5f ) ? "FAIL" : "PASS" );
    }

    printf( "\nMax error: %.2e\n", max_error );
    if ( errors == 0 ) {
        printf( "All samples correct!\n" );
    }
    else {
        printf( "%d errors found\n", errors );
    }
}

// Display performance metrics
void display_metrics( Timings timings, int n, int blockSize )
{
    int numBlocks = ( n + blockSize - 1 ) / blockSize;
    size_t dataSize = n * sizeof( float ) * 3; // 3 arrays (x, y, result)
    float dataSizeMB = dataSize / ( 1024.0f * 1024.0f );

    // Bandwidth calculation (GB/s)
    float bandwidth = ( dataSizeMB / 1024.0f ) / ( timings.kernel_execution / 1000.0f );

    // GFLOPS calculation (2 FLOPs per element: mul + add)
    float gflops = ( n * 2.0f ) / ( timings.kernel_execution * 1e6f );

    printf( "\n" );
    printf( "===========================================\n" );
    printf( "         Performance Metrics\n" );
    printf( "===========================================\n" );
    printf( "Host to Device Transfer:  %8.2f ms\n", timings.memcpy_host_to_device );
    printf( "Kernel Execution:         %8.2f ms\n", timings.kernel_execution );
    printf( "Device to Host Transfer:  %8.2f ms\n", timings.memcpy_device_to_host );
    printf( "-------------------------------------------\n" );
    printf( "Total Time:               %8.2f ms\n", timings.total );
    printf( "===========================================\n" );
    printf( "\n" );
    printf( "===========================================\n" );
    printf( "           Throughput Metrics\n" );
    printf( "===========================================\n" );
    printf( "Array Size:               %d elements\n", n );
    printf( "Data Size:                %.2f MB\n", dataSizeMB );
    printf( "Block Size:               %d\n", blockSize );
    printf( "Number of Blocks:         %d\n", numBlocks );
    printf( "Total Threads:            %d\n", numBlocks * blockSize );
    printf( "Bandwidth:                %.2f GB/s\n", bandwidth );
    printf( "Performance:              %.2f GFLOPS\n", gflops );
    printf( "===========================================\n" );
}

// Print device info
void print_device_info()
{
    int deviceCount;
    CUDA_CHECK( cudaGetDeviceCount( &deviceCount ) );

    if ( deviceCount == 0 ) {
        printf( "No CUDA devices found\n" );
        exit( EXIT_FAILURE );
    }

    int device;
    CUDA_CHECK( cudaGetDevice( &device ) );

    cudaDeviceProp prop;
    CUDA_CHECK( cudaGetDeviceProperties( &prop, device ) );

    printf( "\n" );
    printf( "===========================================\n" );
    printf( "            Device Information\n" );
    printf( "===========================================\n" );
    printf( "Device Name:              %s\n", prop.name );
    printf( "Compute Capability:       %d.%d\n", prop.major, prop.minor );
    printf( "Global Memory:            %.2f GB\n",
            prop.totalGlobalMem / ( 1024.0f * 1024.0f * 1024.0f ) );
    printf( "Shared Memory per Block:  %zu bytes\n", prop.sharedMemPerBlock );
    printf( "Max Threads per Block:    %d\n", prop.maxThreadsPerBlock );
    printf( "Multiprocessors:          %d\n", prop.multiProcessorCount );
    printf( "Clock Rate:               %.2f GHz\n", prop.clockRate / 1e6f );
    printf( "Memory Clock Rate:        %.2f GHz\n", prop.memoryClockRate / 1e6f );
    printf( "Memory Bus Width:         %d-bit\n", prop.memoryBusWidth );
    printf( "===========================================\n" );
}

// Run multiple benchmarks and average
void run_multiple_benchmarks( int n, float a, int num_runs )
{
    printf( "\nRunning %d benchmarks for averaging...\n", num_runs );

    // Allocate and initialize host arrays
    float* h_x = ( float* ) malloc( n * sizeof( float ) );
    float* h_y_original = ( float* ) malloc( n * sizeof( float ) );
    float* h_y = ( float* ) malloc( n * sizeof( float ) );
    float* h_result = ( float* ) malloc( n * sizeof( float ) );

    for ( int i = 0; i < n; i++ ) {
        h_x[i] = i * 1.0f;
        h_y_original[i] = i * 2.0f;
    }

    Timings avg_timings{};

    for ( int run = 0; run < num_runs; run++ ) {
        printf( "Run %d/%d...\n", run + 1, num_runs );

        // Reset y array
        for ( int i = 0; i < n; i++ ) {
            h_y[i] = h_y_original[i];
        }

        Timings timings = run_axpy_cuda( n, a, h_x, h_y, h_result );

        avg_timings.memcpy_host_to_device += timings.memcpy_host_to_device;
        avg_timings.kernel_execution += timings.kernel_execution;
        avg_timings.memcpy_device_to_host += timings.memcpy_device_to_host;
        avg_timings.total += timings.total;
    }

    // Calculate averages
    avg_timings.memcpy_host_to_device /= num_runs;
    avg_timings.kernel_execution /= num_runs;
    avg_timings.memcpy_device_to_host /= num_runs;
    avg_timings.total /= num_runs;

    printf( "\nAveraged results over %d runs:\n", num_runs );
    display_metrics( avg_timings, n, 256 );

    // Cleanup
    free( h_x );
    free( h_y_original );
    free( h_y );
    free( h_result );
}

int main( int argc, char** argv )
{
    // Parse command line arguments
    int n = 10000000; // Default: 10M elements
    float a = 2.5f;
    int num_runs = 1;

    if ( argc > 1 ) {
        n = atoi( argv[1] );
    }
    if ( argc > 2 ) {
        a = atof( argv[2] );
    }
    if ( argc > 3 ) {
        num_runs = atoi( argv[3] );
    }

    printf( "CUDA AXPY Benchmark\n" );
    printf( "===========================================\n" );
    printf( "Array Size (n):           %d\n", n );
    printf( "Scalar (a):               %.2f\n", a );
    printf( "Number of runs:           %d\n", num_runs );

    // Print device info
    print_device_info();

    if ( num_runs > 1 ) {
        run_multiple_benchmarks( n, a, num_runs );
    }
    else {
        // Allocate and initialize host arrays
        float* h_x = ( float* ) malloc( n * sizeof( float ) );
        float* h_y_original = ( float* ) malloc( n * sizeof( float ) );
        float* h_y = ( float* ) malloc( n * sizeof( float ) );
        float* h_result = ( float* ) malloc( n * sizeof( float ) );

        if ( !h_x || !h_y_original || !h_y || !h_result ) {
            fprintf( stderr, "Failed to allocate host memory\n" );
            exit( EXIT_FAILURE );
        }

        // Initialize arrays
        for ( int i = 0; i < n; i++ ) {
            h_x[i] = i * 1.0f;
            h_y_original[i] = i * 2.0f;
            h_y[i] = h_y_original[i];
        }

        // Run benchmark
        printf( "\nRunning benchmark...\n" );
        Timings timings = run_axpy_cuda( n, a, h_x, h_y, h_result );

        // Display results
        display_metrics( timings, n, 256 );
        validate_results( h_x, h_y_original, h_result, a, n );

        // Cleanup
        free( h_x );
        free( h_y_original );
        free( h_y );
        free( h_result );
    }

    CUDA_CHECK( cudaDeviceReset() );

    return 0;
}
