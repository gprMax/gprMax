#ifndef XPU_UPDATE_H
#define XPU_UPDATE_H

#include "common.h"

class xpu_update {
public:
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex, Ey, Ez, Hx, Hy, Hz;
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE, updatecoeffsH;
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID;
    py::detail::unchecked_mutable_reference<float, 3L> Ex_, Ey_, Ez_, Hx_, Hy_, Hz_;
    py::detail::unchecked_reference<float, 2L> updatecoeffsE_, updatecoeffsH_;
    py::detail::unchecked_reference<uint32_t, 4L> ID_;
    int xmin,xmax,ymin,ymax,zmin,zmax;
    int source_xcoord, source_ycoord, source_zcoord;
    float source_start, source_stop;
    py::array_t<float, py::array::c_style | py::array::forcecast> source_waveformvalues_halfdt;
    py::detail::unchecked_reference<float, 1L> source_waveformvalues_halfdt_;
    float source_dl;
    int source_id;
    std::string source_polarization;
    float grid_dt, grid_dx, grid_dy, grid_dz;
    // std::vector<std::vector<std::vector<int>>> timestep_Ex, timestep_Ey, timestep_Ez, timestep_Hx, timestep_Hy, timestep_Hz;
    xpu_update(
        py::array_t<float, py::array::c_style | py::array::forcecast> Ex, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Ey, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Ez, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Hx, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Hy, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Hz, 
        py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE, 
        py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID_,
        int xmin_, int xmax_, int ymin_, int ymax_, int zmin_, int zmax_,

        int source_xcoord, int source_ycoord, int source_zcoord, float source_start, float source_stop,
        py::array_t<float, py::array::c_style | py::array::forcecast> source_waveformvalues_halfdt,
        float source_dl, int source_id, std::string source_polarization,
        float grid_dt, float grid_dx, float grid_dy, float grid_dz
    ) 
        : 
        Ex(Ex), Ey(Ey), Ez(Ez), Hx(Hx), Hy(Hy), Hz(Hz), 
        updatecoeffsE(updatecoeffsE), updatecoeffsH(updatecoeffsH), ID(ID_),
        Ex_(Ex.mutable_unchecked<3>()), Ey_(Ey.mutable_unchecked<3>()), Ez_(Ez.mutable_unchecked<3>()),
        Hx_(Hx.mutable_unchecked<3>()), Hy_(Hy.mutable_unchecked<3>()), Hz_(Hz.mutable_unchecked<3>()),
        updatecoeffsE_(updatecoeffsE.unchecked<2>()), updatecoeffsH_(updatecoeffsH.unchecked<2>()),
        ID_(ID.unchecked<4>()),
        xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), zmin(zmin_), zmax(zmax_),
        source_xcoord(source_xcoord), source_ycoord(source_ycoord), source_zcoord(source_zcoord), source_start(source_start), source_stop(source_stop),
        source_waveformvalues_halfdt(source_waveformvalues_halfdt),
        source_dl(source_dl), source_id(source_id), source_polarization(source_polarization),
        source_waveformvalues_halfdt_(source_waveformvalues_halfdt.unchecked<1>()),
        grid_dt(grid_dt), grid_dx(grid_dx), grid_dy(grid_dy), grid_dz(grid_dz)
        {
            // timestep_Ex = std::vector<std::vector<std::vector<int>>>(xmax-xmin, std::vector<std::vector<int>>(ymax-ymin, std::vector<int>(zmax-zmin, 0))); 
            // timestep_Ey = std::vector<std::vector<std::vector<int>>>(xmax-xmin, std::vector<std::vector<int>>(ymax-ymin, std::vector<int>(zmax-zmin, 0)));
            // timestep_Ez = std::vector<std::vector<std::vector<int>>>(xmax-xmin, std::vector<std::vector<int>>(ymax-ymin, std::vector<int>(zmax-zmin, 0)));
            // timestep_Hx = std::vector<std::vector<std::vector<int>>>(xmax-xmin, std::vector<std::vector<int>>(ymax-ymin, std::vector<int>(zmax-zmin, 0)));
            // timestep_Hy = std::vector<std::vector<std::vector<int>>>(xmax-xmin, std::vector<std::vector<int>>(ymax-ymin, std::vector<int>(zmax-zmin, 0)));
            // timestep_Hz = std::vector<std::vector<std::vector<int>>>(xmax-xmin, std::vector<std::vector<int>>(ymax-ymin, std::vector<int>(zmax-zmin, 0)));
            // Ex_ = Ex.mutable_unchecked<3>();
            // Ey_ = Ey.mutable_unchecked<3>();
            // Ez_ = Ez.mutable_unchecked<3>();
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