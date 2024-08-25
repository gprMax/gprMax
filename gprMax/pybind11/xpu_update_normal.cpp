#include "inc/xpu_update.h"

void xpu_update::update_electric_normal(update_range_t update_range) {
    auto x_range = std::get<0>(update_range);
    auto y_range = std::get<1>(update_range);
    auto z_range = std::get<2>(update_range);

    if (x_range.first == 0) x_range.first = 1;
    if (y_range.first == 0) y_range.first = 1;
    if (z_range.first == 0) z_range.first = 1;

    auto index = [&](int i, int j, int k) {
        return i * shape[1] * shape[2] + j * shape[2] + k;
    };

    auto index_ID = [&](int n, int i, int j, int k) {
        return n * shape[0] * shape[1] * shape[2] + i * shape[1] * shape[2] + j * shape[2] + k;
    };

    for (int i = x_range.first; i < x_range.second; i++) {
        for (int j = y_range.first; j < y_range.second; j++) {
            for (int k = z_range.first; k < z_range.second; k++) {
                int materialEx = ID_[index_ID(0, i, j, k)];
                int materialEy = ID_[index_ID(1, i, j, k)];
                int materialEz = ID_[index_ID(2, i, j, k)];

                Ex_[index(i, j, k)] = (updatecoeffsE_[materialEx * 5 + 0] * Ex_[index(i, j, k)] +
                                       updatecoeffsE_[materialEx * 5 + 2] * (Hz_[index(i, j, k)] - Hz_[index(i, j - 1, k)]) -
                                       updatecoeffsE_[materialEx * 5 + 3] * (Hy_[index(i, j, k)] - Hy_[index(i, j, k - 1)]));

                Ey_[index(i, j, k)] = (updatecoeffsE_[materialEy * 5 + 0] * Ey_[index(i, j, k)] +
                                       updatecoeffsE_[materialEy * 5 + 3] * (Hx_[index(i, j, k)] - Hx_[index(i, j, k - 1)]) -
                                       updatecoeffsE_[materialEy * 5 + 1] * (Hz_[index(i, j, k)] - Hz_[index(i - 1, j, k)]));

                Ez_[index(i, j, k)] = (updatecoeffsE_[materialEz * 5 + 0] * Ez_[index(i, j, k)] +
                                       updatecoeffsE_[materialEz * 5 + 1] * (Hy_[index(i, j, k)] - Hy_[index(i - 1, j, k)]) -
                                       updatecoeffsE_[materialEz * 5 + 2] * (Hx_[index(i, j, k)] - Hx_[index(i, j - 1, k)]));
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

    auto index = [&](int i, int j, int k) {
        return i * shape[1] * shape[2] + j * shape[2] + k;
    };

    auto index_ID = [&](int n, int i, int j, int k) {
        return n * shape[0] * shape[1] * shape[2] + i * shape[1] * shape[2] + j * shape[2] + k;
    };

    for (int i = x_range.first; i < x_range.second; i++) {
        for (int j = y_range.first; j < y_range.second; j++) {
            for (int k = z_range.first; k < z_range.second; k++) {
                int materialHx = ID_[index_ID(3, i, j, k)];
                int materialHy = ID_[index_ID(4, i, j, k)];
                int materialHz = ID_[index_ID(5, i, j, k)];

                Hx_[index(i, j, k)] = (updatecoeffsH_[materialHx * 5 + 0] * Hx_[index(i, j, k)] -
                                       updatecoeffsH_[materialHx * 5 + 2] * (Ez_[index(i, j + 1, k)] - Ez_[index(i, j, k)]) +
                                       updatecoeffsH_[materialHx * 5 + 3] * (Ey_[index(i, j, k + 1)] - Ey_[index(i, j, k)]));

                Hy_[index(i, j, k)] = (updatecoeffsH_[materialHy * 5 + 0] * Hy_[index(i, j, k)] -
                                       updatecoeffsH_[materialHy * 5 + 3] * (Ex_[index(i, j, k + 1)] - Ex_[index(i, j, k)]) +
                                       updatecoeffsH_[materialHy * 5 + 1] * (Ez_[index(i + 1, j, k)] - Ez_[index(i, j, k)]));

                Hz_[index(i, j, k)] = (updatecoeffsH_[materialHz * 5 + 0] * Hz_[index(i, j, k)] -
                                       updatecoeffsH_[materialHz * 5 + 1] * (Ey_[index(i + 1, j, k)] - Ey_[index(i, j, k)]) +
                                       updatecoeffsH_[materialHz * 5 + 2] * (Ex_[index(i, j + 1, k)] - Ex_[index(i, j, k)]));
            }
        }
    }
}

    