#include "inc/xpu_update.h"

void xpu_update::update_electric_source(int current_timestep, update_range_t update_range){
    // do nothing
    auto x_range = std::get<0>(update_range);
    auto y_range = std::get<1>(update_range);
    auto z_range = std::get<2>(update_range);
    // if x_range[0] <= x_index < x_range[1] and y_range[0] <= y_index < y_range[1] and z_range[0] <= z_index < z_range[1]:
    if (
        x_range.first <= source_xcoord && source_xcoord < x_range.second && 
        y_range.first <= source_ycoord && source_ycoord < y_range.second && 
        z_range.first <= source_zcoord && source_zcoord < z_range.second)
        {
        if(current_timestep * grid_dt >= source_start && current_timestep * grid_dt <= source_stop){
            // auto source_waveformvalues_halfdt_ = source_waveformvalues_halfdt.unchecked<1>();
            // auto updatecoeffsE_ = updatecoeffsE.unchecked<2>();
            // auto ID_ = ID.unchecked<4>();
            // auto Ex_ = Ex.mutable_unchecked<3>();
            // auto Ey_ = Ey.mutable_unchecked<3>();
            // auto Ez_ = Ez.mutable_unchecked<3>();
            if (source_polarization == "x"){
                // float id_debug = ID_(source_id, source_xcoord, source_ycoord, source_zcoord);
                // float updatecoeffsE_debug = updatecoeffsE_(ID_(source_id, source_xcoord, source_ycoord, source_zcoord), 4);
                // float source_waveformvalues_halfdt_debug = source_waveformvalues_halfdt_(current_timestep);
                // float grid_dx_debug = grid_dx;
                // float grid_dy_debug = grid_dy;
                // float grid_dz_debug = grid_dz;
                // float Ex_debug = Ex_(source_xcoord, source_ycoord, source_zcoord);
                // float result = (
                //     updatecoeffsE_(ID_(source_id, source_xcoord, source_ycoord, source_zcoord), 4)
                //     * source_waveformvalues_halfdt_(current_timestep)
                //     * source_dl
                //     * (1 / (grid_dx * grid_dy * grid_dz))
                // );
                Ex_(source_xcoord, source_ycoord, source_zcoord) -= (
                    updatecoeffsE_(ID_(source_id, source_xcoord, source_ycoord, source_zcoord), 4)
                    * source_waveformvalues_halfdt_(current_timestep)
                    * source_dl
                    * (1 / (grid_dx * grid_dy * grid_dz))
                );
            }
            else if (source_polarization == "y"){
                Ey_(source_xcoord, source_ycoord, source_zcoord) -= (
                    updatecoeffsE_(ID_(source_id, source_xcoord, source_ycoord, source_zcoord), 4)
                    * source_waveformvalues_halfdt_(current_timestep)
                    * source_dl
                    * (1 / (grid_dx * grid_dy * grid_dz))
                );
            }
            else if (source_polarization == "z"){
                Ez_(source_xcoord, source_ycoord, source_zcoord) -= (
                    updatecoeffsE_(ID_(source_id, source_xcoord, source_ycoord, source_zcoord), 4)
                    * source_waveformvalues_halfdt_(current_timestep)
                    * source_dl
                    * (1 / (grid_dx * grid_dy * grid_dz))
                );
            }
        }    
    }
    return;
}