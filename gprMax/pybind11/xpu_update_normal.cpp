#include "inc/xpu_update.h"

void xpu_update::update_electric_normal(update_range_t update_range) {
    auto x_range = std::get<0>(update_range);
    auto y_range = std::get<1>(update_range);
    auto z_range = std::get<2>(update_range);

    if (x_range.first == 0) x_range.first = 1;
    if (y_range.first == 0) y_range.first = 1;
    if (z_range.first == 0) z_range.first = 1;

    for (int i = x_range.first; i < x_range.second; i++) {
        for (int j = y_range.first; j < y_range.second; j++) {
            for (int k = z_range.first; k < z_range.second; k++) {
                int materialEx = ID_[INDEX_ID(0, i, j, k)];
                Ex_[INDEX(i, j, k)] = (updatecoeffsE_[materialEx * 5 + 0] * Ex_[INDEX(i, j, k)] +
                                       updatecoeffsE_[materialEx * 5 + 2] * (Hz_[INDEX(i, j, k)] - Hz_[INDEX(i, j - 1, k)]) -
                                       updatecoeffsE_[materialEx * 5 + 3] * (Hy_[INDEX(i, j, k)] - Hy_[INDEX(i, j, k - 1)]));
            }
            for (int k = z_range.first; k < z_range.second; k++) {
                int materialEy = ID_[INDEX_ID(1, i, j, k)];
                Ey_[INDEX(i, j, k)] = (updatecoeffsE_[materialEy * 5 + 0] * Ey_[INDEX(i, j, k)] +
                                       updatecoeffsE_[materialEy * 5 + 3] * (Hx_[INDEX(i, j, k)] - Hx_[INDEX(i, j, k - 1)]) -
                                       updatecoeffsE_[materialEy * 5 + 1] * (Hz_[INDEX(i, j, k)] - Hz_[INDEX(i - 1, j, k)]));
            }
            for (int k = z_range.first; k < z_range.second; k++) {
                int materialEz = ID_[INDEX_ID(2, i, j, k)];
                Ez_[INDEX(i, j, k)] = (updatecoeffsE_[materialEz * 5 + 0] * Ez_[INDEX(i, j, k)] +
                                       updatecoeffsE_[materialEz * 5 + 1] * (Hy_[INDEX(i, j, k)] - Hy_[INDEX(i - 1, j, k)]) -
                                       updatecoeffsE_[materialEz * 5 + 2] * (Hx_[INDEX(i, j, k)] - Hx_[INDEX(i, j - 1, k)]));
            }
        }
    }
}


void xpu_update::update_magnetic_normal(update_range_t update_range) {
    auto x_range = std::get<0>(update_range);
    auto y_range = std::get<1>(update_range);
    auto z_range = std::get<2>(update_range);

    if (x_range.second == xmax) x_range.second = xmax - 1;
    if (y_range.second == ymax) y_range.second = ymax - 1;
    if (z_range.second == zmax) z_range.second = zmax - 1;

    for (int i = x_range.first; i < x_range.second; i++) {
        for (int j = y_range.first; j < y_range.second; j++) {
            for (int k = z_range.first; k < z_range.second; k++) {
                int materialHx = ID_[INDEX_ID(3, i, j, k)];
                Hx_[INDEX(i, j, k)] = (updatecoeffsH_[materialHx * 5 + 0] * Hx_[INDEX(i, j, k)] -
                                       updatecoeffsH_[materialHx * 5 + 2] * (Ez_[INDEX(i, j + 1, k)] - Ez_[INDEX(i, j, k)]) +
                                       updatecoeffsH_[materialHx * 5 + 3] * (Ey_[INDEX(i, j, k + 1)] - Ey_[INDEX(i, j, k)]));
            }
            for (int k = z_range.first; k < z_range.second; k++) {
                int materialHy = ID_[INDEX_ID(4, i, j, k)];
                Hy_[INDEX(i, j, k)] = (updatecoeffsH_[materialHy * 5 + 0] * Hy_[INDEX(i, j, k)] -
                                       updatecoeffsH_[materialHy * 5 + 3] * (Ex_[INDEX(i, j, k + 1)] - Ex_[INDEX(i, j, k)]) +
                                       updatecoeffsH_[materialHy * 5 + 1] * (Ez_[INDEX(i + 1, j, k)] - Ez_[INDEX(i, j, k)]));
            }
            for (int k = z_range.first; k < z_range.second; k++) {
                int materialHz = ID_[INDEX_ID(5, i, j, k)];
                Hz_[INDEX(i, j, k)] = (updatecoeffsH_[materialHz * 5 + 0] * Hz_[INDEX(i, j, k)] -
                                       updatecoeffsH_[materialHz * 5 + 1] * (Ey_[INDEX(i + 1, j, k)] - Ey_[INDEX(i, j, k)]) +
                                       updatecoeffsH_[materialHz * 5 + 2] * (Ex_[INDEX(i, j + 1, k)] - Ex_[INDEX(i, j, k)]));
            }
        }
    }
}

    