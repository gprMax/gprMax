#include "inc/xpu_solver.h"

std::pair<int, int> xpu_solver::GetRange(const std::string& shape, const std::string& electric_or_magnetic, int time_block_size, int space_block_size, int sub_timestep, int tile_index, int start, int end) {
    auto between = [](int first, int second, int min, int max) {
        auto clamp = [](int val, int min, int max) {
            return std::max(min, std::min(val, max));
        };
        first = clamp(first, min, max);
        second = clamp(second, min, max);
        return std::make_pair(first, second);
    };

    int start_idx, end_idx;

    if (shape == "m") {
        if (electric_or_magnetic == "E") {
            if (tile_index == 0) {
                start_idx = start;
                end_idx = start_idx + time_block_size + space_block_size - sub_timestep;
                return between(start_idx, end_idx, start, end);
            } else {
                start_idx = start + 2 * space_block_size + time_block_size + sub_timestep + (tile_index - 1) * (2 * space_block_size + 2 * time_block_size - 1);
                end_idx = start_idx + space_block_size + 2 * time_block_size - 1 - 2 * sub_timestep;
                return between(start_idx, end_idx, start, end);
            }
        } else if (electric_or_magnetic == "H") {
            if (tile_index == 0) {
                start_idx = start;
                end_idx = start_idx + time_block_size + space_block_size - 1 - sub_timestep;
                return between(start_idx, end_idx, start, end);
            } else {
                start_idx = start + 2 * space_block_size + time_block_size + sub_timestep + (tile_index - 1) * (2 * space_block_size + 2 * time_block_size - 1);
                end_idx = start_idx + space_block_size + 2 * time_block_size - 2 - 2 * sub_timestep;
                return between(start_idx, end_idx, start, end);
            }
        }
    } else if (shape == "v") {
        if (electric_or_magnetic == "E") {
            if (tile_index == 0) {
                start_idx = start + time_block_size + space_block_size - sub_timestep;
                end_idx = start_idx + space_block_size + 2 * sub_timestep;
                return between(start_idx, end_idx, start, end);
            } else {
                start_idx = start + 2 * space_block_size + time_block_size + sub_timestep + (tile_index - 1) * (2 * space_block_size + 2 * time_block_size - 1);
                start_idx = start_idx + space_block_size + 2 * time_block_size - 1 - 2 * sub_timestep;
                end_idx = start_idx + space_block_size + 2 * sub_timestep;
                return between(start_idx, end_idx, start, end);
            }
        } else if (electric_or_magnetic == "H") {
            if (tile_index == 0) {
                start_idx = start + time_block_size + space_block_size - 1 - sub_timestep;
                end_idx = start_idx + space_block_size + 1 + 2 * sub_timestep;
                return between(start_idx, end_idx, start, end);
            } else {
                start_idx = start + 2 * space_block_size + time_block_size + sub_timestep + (tile_index - 1) * (2 * space_block_size + 2 * time_block_size - 1);
                start_idx = start_idx + space_block_size + 2 * time_block_size - 2 - 2 * sub_timestep;
                end_idx = start_idx + space_block_size + 1 + 2 * sub_timestep;
                return between(start_idx, end_idx, start, end);
            }
        }
    } else if (shape == "p") {
        if (electric_or_magnetic == "E") {
            start_idx = start + tile_index * space_block_size - sub_timestep;
            end_idx = start_idx + space_block_size;
            return between(start_idx, end_idx, start, end);
        } else if (electric_or_magnetic == "H") {
            start_idx = start + tile_index * space_block_size - sub_timestep - 1;
            end_idx = start_idx + space_block_size;
            return between(start_idx, end_idx, start, end);
        }
    }

    // Default return if none of the conditions match
    return std::make_pair(start, end);
}


void xpu_solver::update(int timestep) {
    for (int phase = 0; phase < max_phase; phase++) {
        for (int xx = 0; xx < x_ntiles; xx++) {
            for (int yy = 0; yy < y_ntiles; yy++) {
                for (int zz = 0; zz < z_ntiles; zz++) {
                    for (int t = 0; t < BLT; t++) {
                        int current_timestep = timestep + t;
                        auto x_range_E = GetRange(TX_Tile_Shapes[phase], "E", BLT, BLX, t, xx, xmin, xmax);
                        auto y_range_E = GetRange(TY_Tile_Shapes[phase], "E", BLT, BLX, t, yy, ymin, ymax);
                        auto z_range_E = GetRange(TZ_Tile_Shapes[phase], "E", BLT, BLX, t, zz, zmin, zmax);
                        update_range_t update_range_E = std::make_tuple(x_range_E, y_range_E, z_range_E);
                        xpu_update_instance.update_electric_tile(current_timestep, update_range_E);

                        auto x_range_H = GetRange(TX_Tile_Shapes[phase], "H", BLT, BLX, t, xx, xmin, xmax);
                        auto y_range_H = GetRange(TY_Tile_Shapes[phase], "H", BLT, BLX, t, yy, ymin, ymax);
                        auto z_range_H = GetRange(TZ_Tile_Shapes[phase], "H", BLT, BLX, t, zz, zmin, zmax);
                        update_range_t update_range_H = std::make_tuple(x_range_H, y_range_H, z_range_H);
                        xpu_update_instance.update_magnetic_tile(current_timestep, update_range_H);
                    }
                }
            }
        }
    }
}

// PYBIND11_MODULE(pybind11_xpu_solver, m) {
//     py::class_<xpu_solver>(m, "xpu_solver")
//         .def(py::init<>())
//         .def("init", &xpu_solver::init)
//         .def("update", &xpu_solver::update);
// }

PYBIND11_MODULE(pybind11_xpu_solver, m) {
    py::class_<xpu_solver>(m, "xpu_solver")
        .def(py::init<
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast>,
            int, int, int, int,
            int, int, int,
            int, int, int, int, int, int,
            int,
            std::string, std::string, std::string,
            std::vector<std::string>, std::vector<std::string>, std::vector<std::string>
        >())
        .def("update", &xpu_solver::update);
}