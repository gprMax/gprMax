#ifndef COMMON_H
#define COMMON_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <algorithm>

namespace py = pybind11;

using update_range_t = std::tuple<std::pair<int, int>, std::pair<int, int>, std::pair<int, int>>;

#endif // COMMON_H