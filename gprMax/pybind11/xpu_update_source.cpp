#include "inc/xpu_update.h"

void xpu_update::update_electric_source(int current_timestep, update_range_t update_range) {
    auto x_range = std::get<0>(update_range);
    auto y_range = std::get<1>(update_range);
    auto z_range = std::get<2>(update_range);

    if (x_range.first <= source_xcoord && source_xcoord < x_range.second && 
        y_range.first <= source_ycoord && source_ycoord < y_range.second && 
        z_range.first <= source_zcoord && source_zcoord < z_range.second)
    {
        if (current_timestep * grid_dt >= source_start && current_timestep * grid_dt <= source_stop) {
            int material = ID_[INDEX_ID(source_id, source_xcoord, source_ycoord, source_zcoord)];
            float source_value = updatecoeffsE_[material * 5 + 4] * 
                                 source_waveformvalues_halfdt_[current_timestep] * 
                                 source_dl * 
                                 (1 / (grid_dx * grid_dy * grid_dz));

            if (source_polarization == "x") {
                Ex_[INDEX(source_xcoord, source_ycoord, source_zcoord)] -= source_value;
            }
            else if (source_polarization == "y") {
                Ey_[INDEX(source_xcoord, source_ycoord, source_zcoord)] -= source_value;
            }
            else if (source_polarization == "z") {
                Ez_[INDEX(source_xcoord, source_ycoord, source_zcoord)] -= source_value;
            }
        }
    }
}
