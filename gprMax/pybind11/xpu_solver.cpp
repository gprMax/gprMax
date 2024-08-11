#include "xpu_solver.h"

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
    // translate the following python code to C++:
    // for phase in range(max_phase):
    // for xx in range(x_ntiles):
    //     for yy in range(y_ntiles):
    //         for zz in range(z_ntiles):
    //             for t in range(BLT):
    //                 current_timestep = tt + t
    //                 x_range=GetRange(TX_Tile_Shapes[phase], "E", BLT, BLX, t, xx, xmin, xmax)
    //                 y_range=GetRange(TY_Tile_Shapes[phase], "E", BLT, BLX, t, yy, ymin, ymax)
    //                 z_range=GetRange(TZ_Tile_Shapes[phase], "E", BLT, BLX, t, zz, zmin, zmax)
    //                 update_range = (x_range, y_range, z_range)
    //                 self.update_electric_tile(update_range, current_timestep)
    //                 x_range=GetRange(TX_Tile_Shapes[phase], "H", BLT, BLX, t, xx, xmin, xmax)
    //                 y_range=GetRange(TY_Tile_Shapes[phase], "H", BLT, BLX, t, yy, ymin, ymax)
    //                 z_range=GetRange(TZ_Tile_Shapes[phase], "H", BLT, BLX, t, zz, zmin, zmax)
    //                 update_range = (x_range, y_range, z_range)
    //                 self.update_magnetic_tile(update_range, current_timestep)
    for (int phase = 0; phase < max_phase; phase++) {
        for (int xx = 0; xx < x_ntiles; xx++) {
            for (int yy = 0; yy < y_ntiles; yy++) {
                for (int zz = 0; zz < z_ntiles; zz++) {
                    for (int t = 0; t < BLT; t++) {
                        int current_timestep = timestep + t;
                        auto x_range = GetRange(TX_Tile_Shapes[phase], "E", BLT, BLX, t, xx, xmin, xmax);
                        auto y_range = GetRange(TY_Tile_Shapes[phase], "E", BLT, BLX, t, yy, ymin, ymax);
                        auto z_range = GetRange(TZ_Tile_Shapes[phase], "E", BLT, BLX, t, zz, zmin, zmax);
                        auto update_range_E = std::make_tuple(x_range, y_range, z_range);
                        std::cout << "update_range_E: " 
                        << std::get<0>(update_range_E).first << ", " << std::get<0>(update_range_E).second << ", " 
                        << std::get<1>(update_range_E).first << ", " << std::get<1>(update_range_E).second << ", " 
                        << std::get<2>(update_range_E).first << ", " << std::get<2>(update_range_E).second << std::endl;

                        x_range = GetRange(TX_Tile_Shapes[phase], "H", BLT, BLX, t, xx, xmin, xmax);
                        y_range = GetRange(TY_Tile_Shapes[phase], "H", BLT, BLX, t, yy, ymin, ymax);
                        z_range = GetRange(TZ_Tile_Shapes[phase], "H", BLT, BLX, t, zz, zmin, zmax);
                        auto update_range_H = std::make_tuple(x_range, y_range, z_range);
                        std::cout << "update_range_H: "
                        << std::get<0>(update_range_H).first << ", " << std::get<0>(update_range_H).second << ", "
                        << std::get<1>(update_range_H).first << ", " << std::get<1>(update_range_H).second << ", "
                        << std::get<2>(update_range_H).first << ", " << std::get<2>(update_range_H).second << std::endl;
                    }
                }
            }
        }
    }
}

PYBIND11_MODULE(pybind11_xpu_solver, m) {
    py::class_<xpu_solver>(m, "xpu_solver")
        .def(py::init<>())
        .def("init", &xpu_solver::init)
        .def("update", &xpu_solver::update);
}