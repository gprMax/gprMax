#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <algorithm>

namespace py = pybind11;

class xpu_solver {
public:
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex, Ey, Ez, Hx, Hy, Hz;
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE, updatecoeffsH;
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID;
    int BLT,BLX,BLY,BLZ;
    int x_ntiles, y_ntiles, z_ntiles;
    int xmin,xmax,ymin,ymax,zmin,zmax;
    int max_phase;
    std::string tx_tiling_type, ty_tiling_type, tz_tiling_type;
    std::vector<std::string> TX_Tile_Shapes, TY_Tile_Shapes, TZ_Tile_Shapes;
    void init(
        py::array_t<float, py::array::c_style | py::array::forcecast> Ex_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Ey_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Ez_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Hx_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Hy_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Hz_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH_,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID_,
        int BLT_, int BLX_, int BLY_, int BLZ_,
        int x_ntiles_, int y_ntiles_, int z_ntiles_,
        int xmin_, int xmax_, int ymin_, int ymax_, int zmin_, int zmax_, 
        int max_phase_,
        std::string tx_tiling_type_, std::string ty_tiling_type_, std::string tz_tiling_type_, 
        std::vector<std::string> TX_Tile_Shapes_, std::vector<std::string> TY_Tile_Shapes_, std::vector<std::string> TZ_Tile_Shapes_
        ) {

        Ex = Ex_;
        Ey = Ey_;
        Ez = Ez_;
        Hx = Hx_;
        Hy = Hy_;
        Hz = Hz_;
        updatecoeffsE = updatecoeffsE_;
        updatecoeffsH = updatecoeffsH_;
        ID = ID_;
        BLT = BLT_, BLX = BLX_, BLY = BLY_, BLZ = BLZ_;
        x_ntiles = x_ntiles_, y_ntiles = y_ntiles_, z_ntiles = z_ntiles_;
        xmin = xmin_, xmax = xmax_, ymin = ymin_, ymax = ymax_, zmin = zmin_, zmax = zmax_;
        max_phase = max_phase_;
        tx_tiling_type = tx_tiling_type_, ty_tiling_type = ty_tiling_type_, tz_tiling_type = tz_tiling_type_;
        TX_Tile_Shapes = TX_Tile_Shapes_, TY_Tile_Shapes = TY_Tile_Shapes_, TZ_Tile_Shapes = TZ_Tile_Shapes_;
    }

    std::pair<int, int> GetRange(
        const std::string& shape, 
        const std::string& electric_or_magnetic, 
        int time_block_size, 
        int space_block_size, 
        int sub_timestep, 
        int tile_index, 
        int start, 
        int end
    );

    void update(int timestep);
};

