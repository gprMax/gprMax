#include "inc/xpu_update.h"

void xpu_update::update_electric_source(int current_timestep, update_range_t update_range){
    // auto x_range = std::get<0>(update_range);
    // auto y_range = std::get<1>(update_range);
    // auto z_range = std::get<2>(update_range);
    // if (
    //     x_range.first <= source_xcoord && source_xcoord < x_range.second && 
    //     y_range.first <= source_ycoord && source_ycoord < y_range.second && 
    //     z_range.first <= source_zcoord && source_zcoord < z_range.second)
    //     {
    //     if(current_timestep * grid_dt >= source_start && current_timestep * grid_dt <= source_stop){
    //         if (source_polarization == "x"){
    //             Ex_(source_xcoord, source_ycoord, source_zcoord) -= (
    //                 updatecoeffsE_(ID_(source_id, source_xcoord, source_ycoord, source_zcoord), 4)
    //                 * source_waveformvalues_halfdt_(current_timestep)
    //                 * source_dl
    //                 * (1 / (grid_dx * grid_dy * grid_dz))
    //             );
    //         }
    //         else if (source_polarization == "y"){
    //             Ey_(source_xcoord, source_ycoord, source_zcoord) -= (
    //                 updatecoeffsE_(ID_(source_id, source_xcoord, source_ycoord, source_zcoord), 4)
    //                 * source_waveformvalues_halfdt_(current_timestep)
    //                 * source_dl
    //                 * (1 / (grid_dx * grid_dy * grid_dz))
    //             );
    //         }
    //         else if (source_polarization == "z"){
    //             Ez_(source_xcoord, source_ycoord, source_zcoord) -= (
    //                 updatecoeffsE_(ID_(source_id, source_xcoord, source_ycoord, source_zcoord), 4)
    //                 * source_waveformvalues_halfdt_(current_timestep)
    //                 * source_dl
    //                 * (1 / (grid_dx * grid_dy * grid_dz))
    //             );
    //         }
    //     }    
    // }
    return;
}