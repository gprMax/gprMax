#ifndef XPU_SOLVER_H
#define XPU_SOLVER_H

#include "common.h"
#include "xpu_update.h"
#include "utils.h"

namespace py = pybind11;

class xpu_solver {
public:
    int BLT,BLX,BLY,BLZ;
    int x_ntiles, y_ntiles, z_ntiles;
    int xmin,xmax,ymin,ymax,zmin,zmax;
    int max_phase;
    std::string tx_tiling_type, ty_tiling_type, tz_tiling_type;
    std::vector<std::string> TX_Tile_Shapes, TY_Tile_Shapes, TZ_Tile_Shapes;
    xpu_update xpu_update_instance;
    xpu_solver(
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
        std::vector<std::string> TX_Tile_Shapes_, std::vector<std::string> TY_Tile_Shapes_, std::vector<std::string> TZ_Tile_Shapes_,
        
        int source_xcoord, int source_ycoord, int source_zcoord, float source_start, float source_stop,
        py::array_t<float, py::array::c_style | py::array::forcecast> source_waveformvalues_halfdt,
        float source_dl, int source_id,
        float grid_dt, float grid_dx, float grid_dy, float grid_dz
    ) : xpu_update_instance(
            Ex_, Ey_, Ez_, Hx_, Hy_, Hz_, 
            updatecoeffsE_, updatecoeffsH_, ID_, 
            xmin_, xmax_, ymin_, ymax_, zmin_, zmax_,
            source_xcoord, source_ycoord, source_zcoord, source_start, source_stop,
            source_waveformvalues_halfdt,
            source_dl, source_id,
            grid_dt, grid_dx, grid_dy, grid_dz
        ),
        BLT(BLT_), BLX(BLX_), BLY(BLY_), BLZ(BLZ_),
        x_ntiles(x_ntiles_), y_ntiles(y_ntiles_), z_ntiles(z_ntiles_),
        xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), zmin(zmin_), zmax(zmax_),
        max_phase(max_phase_),
        tx_tiling_type(tx_tiling_type_), ty_tiling_type(ty_tiling_type_), tz_tiling_type(tz_tiling_type_),
        TX_Tile_Shapes(TX_Tile_Shapes_), TY_Tile_Shapes(TY_Tile_Shapes_), TZ_Tile_Shapes(TZ_Tile_Shapes_) {}

    void update(int timestep);
};

#endif // XPU_SOLVER_H
