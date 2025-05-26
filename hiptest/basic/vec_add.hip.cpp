#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// GPU kernel
__global__ void vec_add_kernel(const float *a, const float *b, float *c, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        c[i] = a[i] + b[i];
}

// Python-exposed wrapper
py::array_t<float> vector_add(py::array_t<float> a_np, py::array_t<float> b_np) {
    py::buffer_info a_info = a_np.request();
    py::buffer_info b_info = b_np.request();

    if (a_info.size != b_info.size)
        throw std::runtime_error("Input sizes must match!");

    int N = a_info.size;
    const float* a_ptr = static_cast<float*>(a_info.ptr);
    const float* b_ptr = static_cast<float*>(b_info.ptr);

    // Allocate output
    py::array_t<float> c_np(N);
    py::buffer_info c_info = c_np.request();
    float* c_ptr = static_cast<float*>(c_info.ptr);

    // Allocate GPU memory
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);

    hipMemcpy(d_a, a_ptr, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, b_ptr, size, hipMemcpyHostToDevice);

    vec_add_kernel<<<(N + 255)/256, 256>>>(d_a, d_b, d_c, N);
    hipMemcpy(c_ptr, d_c, size, hipMemcpyDeviceToHost);

    hipFree(d_a); hipFree(d_b); hipFree(d_c);

    return c_np;
}

PYBIND11_MODULE(vecadd, m) {
    m.def("vector_add", &vector_add, "Add two float arrays using HIP");
}
