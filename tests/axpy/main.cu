#include <iostream>
#include <vector>
#include <cmath>

__global__ void axpy_kernel(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    const int N = 1 << 20;
    const float A = 2.0f;

    std::vector<float> h_x(N);
    std::vector<float> h_y(N);

    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(N - i);
    }

    float *d_x, *d_y;

    cudaError_t err = cudaMalloc(&d_x, N * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error allocating d_x: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    err = cudaMalloc(&d_y, N * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error allocating d_y: " << cudaGetErrorString(err) << std::endl;
        
        cudaFree(d_x);
        return 1;
    }

    err = cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error copying h_x to d_x: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_x); cudaFree(d_y);
        return 1;
    }

    err = cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error copying h_y to d_y: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_x); cudaFree(d_y);
        return 1;
    }
    
    const int threadsPerBlock = 256;
    
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching AXPY kernel with N=" << N
              << ", A=" << A
              << ", blocksPerGrid=" << blocksPerGrid
              << ", threadsPerBlock=" << threadsPerBlock << std::endl;
    
    axpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, A, d_x, d_y);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during kernel execution: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_x); cudaFree(d_y);
        return 1;
    }
    
    err = cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error copying d_y to h_y: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_x); cudaFree(d_y);
        return 1;
    }

    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected_y_i = A * static_cast<float>(i) + static_cast<float>(N - i);
        if (std::abs(h_y[i] - expected_y_i) > 1e-5) {
            std::cerr << "Mismatch at index " << i
                      << ": Expected " << expected_y_i
                      << ", Got " << h_y[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Verification successful! Results match." << std::endl;
    } else {
        std::cout << "Verification failed!" << std::endl;
    }

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
