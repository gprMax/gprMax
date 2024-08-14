#ifndef XPU_UPDATE_H
#define XPU_UPDATE_H

#include "common.h"

class xpu_update {
public:
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex, Ey, Ez, Hx, Hy, Hz;
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE, updatecoeffsH;
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID;
    int xmin,xmax,ymin,ymax,zmin,zmax;
    int source_xcoord, source_ycoord, source_zcoord;
    float source_start, source_stop;
    py::array_t<float, py::array::c_style | py::array::forcecast> source_waveformvalues_halfdt;
    float source_dl;
    int source_id;
    std::string source_polarization;
    float grid_dt, grid_dx, grid_dy, grid_dz;
    std::vector<std::vector<std::vector<int>>> timestep_Ex, timestep_Ey, timestep_Ez, timestep_Hx, timestep_Hy, timestep_Hz;
    xpu_update(
        py::array_t<float, py::array::c_style | py::array::forcecast> Ex_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Ey_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Ez_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Hx_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Hy_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Hz_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE_, 
        py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH_,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID_,
        int xmin_, int xmax_, int ymin_, int ymax_, int zmin_, int zmax_,

        int source_xcoord, int source_ycoord, int source_zcoord, float source_start, float source_stop,
        py::array_t<float, py::array::c_style | py::array::forcecast> source_waveformvalues_halfdt,
        float source_dl, int source_id, std::string source_polarization,
        float grid_dt, float grid_dx, float grid_dy, float grid_dz
    ) 
        : 
        Ex(Ex_), Ey(Ey_), Ez(Ez_), Hx(Hx_), Hy(Hy_), Hz(Hz_), 
        updatecoeffsE(updatecoeffsE_), updatecoeffsH(updatecoeffsH_), ID(ID_),
        xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), zmin(zmin_), zmax(zmax_),
        source_xcoord(source_xcoord), source_ycoord(source_ycoord), source_zcoord(source_zcoord), source_start(source_start), source_stop(source_stop),
        source_waveformvalues_halfdt(source_waveformvalues_halfdt),
        source_dl(source_dl), source_id(source_id), source_polarization(source_polarization),
        grid_dt(grid_dt), grid_dx(grid_dx), grid_dy(grid_dy), grid_dz(grid_dz)
        {
            timestep_Ex = std::vector<std::vector<std::vector<int>>>(xmax-xmin, std::vector<std::vector<int>>(ymax-ymin, std::vector<int>(zmax-zmin, 0))); 
            timestep_Ey = std::vector<std::vector<std::vector<int>>>(xmax-xmin, std::vector<std::vector<int>>(ymax-ymin, std::vector<int>(zmax-zmin, 0)));
            timestep_Ez = std::vector<std::vector<std::vector<int>>>(xmax-xmin, std::vector<std::vector<int>>(ymax-ymin, std::vector<int>(zmax-zmin, 0)));
            timestep_Hx = std::vector<std::vector<std::vector<int>>>(xmax-xmin, std::vector<std::vector<int>>(ymax-ymin, std::vector<int>(zmax-zmin, 0)));
            timestep_Hy = std::vector<std::vector<std::vector<int>>>(xmax-xmin, std::vector<std::vector<int>>(ymax-ymin, std::vector<int>(zmax-zmin, 0)));
            timestep_Hz = std::vector<std::vector<std::vector<int>>>(xmax-xmin, std::vector<std::vector<int>>(ymax-ymin, std::vector<int>(zmax-zmin, 0)));
        }
    void update_electric_tile(int current_timestep, update_range_t update_range){
        update_electric_normal(update_range);
        update_electric_source(current_timestep, update_range);
    }
    void update_magnetic_tile(int current_timestep, update_range_t update_range){
        update_magnetic_normal(update_range);
    }
    void update_electric_normal(update_range_t update_range);
    void update_magnetic_normal(update_range_t update_range);
    void update_electric_source(int current_timestep, update_range_t update_range);
};

#endif // XPU_UPDATE_H