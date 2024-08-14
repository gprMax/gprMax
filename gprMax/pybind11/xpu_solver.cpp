#include "inc/xpu_solver.h"

void xpu_solver::update(int timestep) {
    for (int phase = 0; phase < max_phase; phase++) {
        for (int xx = 0; xx < x_ntiles; xx++) {
            for (int yy = 0; yy < y_ntiles; yy++) {
                for (int zz = 0; zz < z_ntiles; zz++) {
                    for (int t = 0; t < BLT; t++) {
                        int current_timestep = timestep + t;
                        auto x_range_E = GetRange(TX_Tile_Shapes[phase], "E", BLT, BLX, t, xx, xmin, xmax);
                        auto y_range_E = GetRange(TY_Tile_Shapes[phase], "E", BLT, BLY, t, yy, ymin, ymax);
                        auto z_range_E = GetRange(TZ_Tile_Shapes[phase], "E", BLT, BLZ, t, zz, zmin, zmax);
                        update_range_t update_range_E = std::make_tuple(x_range_E, y_range_E, z_range_E);
                        xpu_update_instance.update_electric_tile(current_timestep, update_range_E);

                        auto x_range_H = GetRange(TX_Tile_Shapes[phase], "H", BLT, BLX, t, xx, xmin, xmax);
                        auto y_range_H = GetRange(TY_Tile_Shapes[phase], "H", BLT, BLY, t, yy, ymin, ymax);
                        auto z_range_H = GetRange(TZ_Tile_Shapes[phase], "H", BLT, BLZ, t, zz, zmin, zmax);
                        update_range_t update_range_H = std::make_tuple(x_range_H, y_range_H, z_range_H);
                        xpu_update_instance.update_magnetic_tile(current_timestep, update_range_H);
                    }
                }
            }
        }
    }
}

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
            std::vector<std::string>, std::vector<std::string>, std::vector<std::string>,
            int, int, int, float, float,
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            float, int, std::string,
            float, float, float, float
        >())
        .def("update", &xpu_solver::update);
}