#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <iostream>

namespace py = pybind11;

class xpu_solver {
public:
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex, Ey, Ez, Hx, Hy, Hz;
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE, updatecoeffsH;
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID;
    void init(py::array_t<float, py::array::c_style | py::array::forcecast> Ex_, 
              py::array_t<float, py::array::c_style | py::array::forcecast> Ey_, 
              py::array_t<float, py::array::c_style | py::array::forcecast> Ez_, 
              py::array_t<float, py::array::c_style | py::array::forcecast> Hx_, 
              py::array_t<float, py::array::c_style | py::array::forcecast> Hy_, 
              py::array_t<float, py::array::c_style | py::array::forcecast> Hz_, 
              py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE_, 
              py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH_,
              py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID_) {
        Ex = Ex_;
        Ey = Ey_;
        Ez = Ez_;
        Hx = Hx_;
        Hy = Hy_;
        Hz = Hz_;
        updatecoeffsE = updatecoeffsE_;
        updatecoeffsH = updatecoeffsH_;
        ID = ID_;
    }
    void update() {
        // do nothing
        std::cout << "Ex: " << Ex.size() << std::endl;
        std::cout << "Ey: " << Ey.size() << std::endl;
        std::cout << "Ez: " << Ez.size() << std::endl;
        std::cout << "Hx: " << Hx.size() << std::endl;
        std::cout << "Hy: " << Hy.size() << std::endl;
        std::cout << "Hz: " << Hz.size() << std::endl;
    }
};

PYBIND11_MODULE(pybind11_xpu_solver, m) {
    py::class_<xpu_solver>(m, "xpu_solver")
        .def(py::init<>())
        .def("init", &xpu_solver::init)
        .def("update", &xpu_solver::update);
}

