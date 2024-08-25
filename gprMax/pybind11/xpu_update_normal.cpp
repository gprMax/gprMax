#include "inc/xpu_update.h"

void xpu_update::update_electric_normal(update_range_t update_range) {
    auto x_range = std::get<0>(update_range);
    auto y_range = std::get<1>(update_range);
    auto z_range = std::get<2>(update_range);

    if (x_range.first == 0) x_range.first = 1;
    if (y_range.first == 0) y_range.first = 1;
    if (z_range.first == 0) z_range.first = 1;

    // auto ID_info = ID.request();
    // uint32_t *ID_ = static_cast<uint32_t *>(ID_info.ptr);

    // auto Ex_info = Ex.request();
    // float *Ex_ = static_cast<float *>(Ex_info.ptr);

    // auto Ey_info = Ey.request();
    // float *Ey_ = static_cast<float *>(Ey_info.ptr);

    // auto Ez_info = Ez.request();
    // float *Ez_ = static_cast<float *>(Ez_info.ptr);

    // auto Hx_info = Hx.request();
    // float *Hx_ = static_cast<float *>(Hx_info.ptr);

    // auto Hy_info = Hy.request();
    // float *Hy_ = static_cast<float *>(Hy_info.ptr);

    // auto Hz_info = Hz.request();
    // float *Hz_ = static_cast<float *>(Hz_info.ptr);

    // auto updatecoeffsE_info = updatecoeffsE.request();
    // float *updatecoeffsE_ = static_cast<float *>(updatecoeffsE_info.ptr);

    // auto shape = Ex_info.shape;

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

                Ex_[index(i, j, k)] = (updatecoeffsE_[materialEx * 4 + 0] * Ex_[index(i, j, k)] +
                                       updatecoeffsE_[materialEx * 4 + 2] * (Hz_[index(i, j, k)] - Hz_[index(i, j - 1, k)]) -
                                       updatecoeffsE_[materialEx * 4 + 3] * (Hy_[index(i, j, k)] - Hy_[index(i, j, k - 1)]));

                Ey_[index(i, j, k)] = (updatecoeffsE_[materialEy * 4 + 0] * Ey_[index(i, j, k)] +
                                       updatecoeffsE_[materialEy * 4 + 3] * (Hx_[index(i, j, k)] - Hx_[index(i, j, k - 1)]) -
                                       updatecoeffsE_[materialEy * 4 + 1] * (Hz_[index(i, j, k)] - Hz_[index(i - 1, j, k)]));

                Ez_[index(i, j, k)] = (updatecoeffsE_[materialEz * 4 + 0] * Ez_[index(i, j, k)] +
                                       updatecoeffsE_[materialEz * 4 + 1] * (Hy_[index(i, j, k)] - Hy_[index(i - 1, j, k)]) -
                                       updatecoeffsE_[materialEz * 4 + 2] * (Hx_[index(i, j, k)] - Hx_[index(i, j - 1, k)]));
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

    // auto ID_info = ID.request();
    // uint32_t *ID_ = static_cast<uint32_t *>(ID_info.ptr);

    // auto Ex_info = Ex.request();
    // float *Ex_ = static_cast<float *>(Ex_info.ptr);

    // auto Ey_info = Ey.request();
    // float *Ey_ = static_cast<float *>(Ey_info.ptr);

    // auto Ez_info = Ez.request();
    // float *Ez_ = static_cast<float *>(Ez_info.ptr);

    // auto Hx_info = Hx.request();
    // float *Hx_ = static_cast<float *>(Hx_info.ptr);

    // auto Hy_info = Hy.request();
    // float *Hy_ = static_cast<float *>(Hy_info.ptr);

    // auto Hz_info = Hz.request();
    // float *Hz_ = static_cast<float *>(Hz_info.ptr);

    // auto updatecoeffsE_info = updatecoeffsE.request();
    // float *updatecoeffsE_ = static_cast<float *>(updatecoeffsE_info.ptr);

    // auto updatecoeffsH_info = updatecoeffsH.request();
    // float *updatecoeffsH_ = static_cast<float *>(updatecoeffsH_info.ptr);

    // auto shape = Ex_info.shape;

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

                Hx_[index(i, j, k)] = (updatecoeffsH_[materialHx * 4 + 0] * Hx_[index(i, j, k)] -
                                       updatecoeffsH_[materialHx * 4 + 2] * (Ez_[index(i, j + 1, k)] - Ez_[index(i, j, k)]) +
                                       updatecoeffsH_[materialHx * 4 + 3] * (Ey_[index(i, j, k + 1)] - Ey_[index(i, j, k)]));

                Hy_[index(i, j, k)] = (updatecoeffsH_[materialHy * 4 + 0] * Hy_[index(i, j, k)] -
                                       updatecoeffsH_[materialHy * 4 + 3] * (Ex_[index(i, j, k + 1)] - Ex_[index(i, j, k)]) +
                                       updatecoeffsH_[materialHy * 4 + 1] * (Ez_[index(i + 1, j, k)] - Ez_[index(i, j, k)]));

                Hz_[index(i, j, k)] = (updatecoeffsH_[materialHz * 4 + 0] * Hz_[index(i, j, k)] -
                                       updatecoeffsH_[materialHz * 4 + 1] * (Ey_[index(i + 1, j, k)] - Ey_[index(i, j, k)]) +
                                       updatecoeffsH_[materialHz * 4 + 2] * (Ex_[index(i, j + 1, k)] - Ex_[index(i, j, k)]));
            }
        }
    }
}

    