#ifndef XPU_UPDATE_H
#define XPU_UPDATE_H

#include "common.h"

class xpu_update {
public:
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex, Ey, Ez, Hx, Hy, Hz;
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE, updatecoeffsH;
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID;
    uint32_t *ID_;
    float *Ex_, *Ey_, *Ez_, *Hx_, *Hy_, *Hz_, *updatecoeffsE_, *updatecoeffsH_;
    std::vector<pybind11::ssize_t> shape;
    int xmin,xmax,ymin,ymax,zmin,zmax;
    int source_xcoord, source_ycoord, source_zcoord;
    float source_start, source_stop;
    py::array_t<float, py::array::c_style | py::array::forcecast> source_waveformvalues_halfdt;
    float *source_waveformvalues_halfdt_;
    float source_dl;
    int source_id;
    std::string source_polarization;
    float grid_dt, grid_dx, grid_dy, grid_dz;
    xpu_update(
        py::array_t<float, py::array::c_style | py::array::forcecast> Ex, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Ey, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Ez, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Hx, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Hy, 
        py::array_t<float, py::array::c_style | py::array::forcecast> Hz, 
        py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE, 
        py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID,
        int xmin_, int xmax_, int ymin_, int ymax_, int zmin_, int zmax_,

        int source_xcoord, int source_ycoord, int source_zcoord, float source_start, float source_stop,
        py::array_t<float, py::array::c_style | py::array::forcecast> source_waveformvalues_halfdt,
        float source_dl, int source_id, std::string source_polarization,
        float grid_dt, float grid_dx, float grid_dy, float grid_dz
    ) 
        : 
        Ex(Ex), Ey(Ey), Ez(Ez), Hx(Hx), Hy(Hy), Hz(Hz), 
        updatecoeffsE(updatecoeffsE), updatecoeffsH(updatecoeffsH), 
        ID(ID),
        xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), zmin(zmin_), zmax(zmax_),
        source_xcoord(source_xcoord), source_ycoord(source_ycoord), source_zcoord(source_zcoord), source_start(source_start), source_stop(source_stop),
        source_waveformvalues_halfdt(source_waveformvalues_halfdt),
        source_dl(source_dl), source_id(source_id), source_polarization(source_polarization),
        grid_dt(grid_dt), grid_dx(grid_dx), grid_dy(grid_dy), grid_dz(grid_dz)
        {
            auto ID_info = ID.request();
            ID_ = static_cast<uint32_t *>(ID_info.ptr);

            auto Ex_info = Ex.request();
            Ex_ = static_cast<float *>(Ex_info.ptr);

            auto Ey_info = Ey.request();
            Ey_ = static_cast<float *>(Ey_info.ptr);

            auto Ez_info = Ez.request();
            Ez_ = static_cast<float *>(Ez_info.ptr);

            auto Hx_info = Hx.request();
            Hx_ = static_cast<float *>(Hx_info.ptr);

            auto Hy_info = Hy.request();
            Hy_ = static_cast<float *>(Hy_info.ptr);

            auto Hz_info = Hz.request();
            Hz_ = static_cast<float *>(Hz_info.ptr);

            auto updatecoeffsE_info = updatecoeffsE.request();
            updatecoeffsE_ = static_cast<float *>(updatecoeffsE_info.ptr);

            auto updatecoeffsH_info = updatecoeffsH.request();
            updatecoeffsH_ = static_cast<float *>(updatecoeffsH_info.ptr);

            auto source_waveformvalues_halfdt_info = source_waveformvalues_halfdt.request();
            source_waveformvalues_halfdt_ = static_cast<float *>(source_waveformvalues_halfdt_info.ptr);

            shape = ID_info.shape;
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