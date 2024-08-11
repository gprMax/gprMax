#ifndef XPU_UPDATE_H
#define XPU_UPDATE_H

#include "common.h"

class xpu_update {
public:
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex, Ey, Ez, Hx, Hy, Hz;
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE, updatecoeffsH;
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID;
    int xmin,xmax,ymin,ymax,zmin,zmax;
    xpu_update(py::array_t<float, py::array::c_style | py::array::forcecast> Ex_, 
               py::array_t<float, py::array::c_style | py::array::forcecast> Ey_, 
               py::array_t<float, py::array::c_style | py::array::forcecast> Ez_, 
               py::array_t<float, py::array::c_style | py::array::forcecast> Hx_, 
               py::array_t<float, py::array::c_style | py::array::forcecast> Hy_, 
               py::array_t<float, py::array::c_style | py::array::forcecast> Hz_, 
               py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE_, 
               py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH_,
               py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID_,
               int xmin_, int xmax_, int ymin_, int ymax_, int zmin_, int zmax_) 
        : Ex(Ex_), Ey(Ey_), Ez(Ez_), Hx(Hx_), Hy(Hy_), Hz(Hz_), 
          updatecoeffsE(updatecoeffsE_), updatecoeffsH(updatecoeffsH_), ID(ID_),
          xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), zmin(zmin_), zmax(zmax_) {
    }
    void update_electric_tile(int current_timestep, update_range_t update_range){
        update_electric_normal(update_range);
    }
    void update_magnetic_tile(int current_timestep, update_range_t update_range){
        update_magnetic_normal(update_range);
    }
    void update_electric_normal(update_range_t update_range);
    void update_magnetic_normal(update_range_t update_range);
};

#endif // XPU_UPDATE_H