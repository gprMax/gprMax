#include "inc/xpu_update.h"

void xpu_update::update_electric_normal(update_range_t update_range){
    auto updatecoeffsE_ = updatecoeffsE.unchecked<2>();
    auto ID_ = ID.unchecked<4>();
    auto Ex_ = Ex.mutable_unchecked<3>();
    auto Ey_ = Ey.mutable_unchecked<3>();
    auto Ez_ = Ez.mutable_unchecked<3>();
    auto Hx_ = Hx.unchecked<3>();
    auto Hy_ = Hy.unchecked<3>();
    auto Hz_ = Hz.unchecked<3>();
    auto x_range = std::get<0>(update_range);
    auto y_range = std::get<1>(update_range);
    auto z_range = std::get<2>(update_range);
    for(int i = x_range.first; i < x_range.second; i++){
        for(int j = y_range.first; j < y_range.second; j++){
            for(int k = z_range.first; k < z_range.second; k++){
                    int materialEx = ID_(0, i, j, k);
                    int materialEy = ID_(1, i, j, k);
                    int materialEz = ID_(2, i, j, k);
                    if(j != 0 && k != 0)
                        Ex_(i, j, k) = (updatecoeffsE_(materialEx, 0) * Ex_(i, j, k) +
                                        updatecoeffsE_(materialEx, 2) * (Hz_(i, j, k) - Hz_(i, j - 1, k)) -
                                        updatecoeffsE_(materialEx, 3) * (Hy_(i, j, k) - Hy_(i, j, k - 1)));
                    if(i != 0 && k != 0)
                        Ey_(i, j, k) = (updatecoeffsE_(materialEy, 0) * Ey_(i, j, k) +
                                        updatecoeffsE_(materialEy, 3) * (Hx_(i, j, k) - Hx_(i, j, k - 1)) -
                                        updatecoeffsE_(materialEy, 1) * (Hz_(i, j, k) - Hz_(i - 1, j, k)));
                    if(i != 0 && j != 0)
                        Ez_(i, j, k) = (updatecoeffsE_(materialEz, 0) * Ez_(i, j, k) +
                                        updatecoeffsE_(materialEz, 1) * (Hy_(i, j, k) - Hy_(i - 1, j, k)) -
                                        updatecoeffsE_(materialEz, 2) * (Hx_(i, j, k) - Hx_(i, j - 1, k)));
            }
        }
    }
}

void xpu_update::update_magnetic_normal(update_range_t update_range){
    auto updatecoeffsH_ = updatecoeffsH.unchecked<2>();
    auto ID_ = ID.unchecked<4>();
    auto Ex_ = Ex.unchecked<3>();
    auto Ey_ = Ey.unchecked<3>();
    auto Ez_ = Ez.unchecked<3>();
    auto Hx_ = Hx.mutable_unchecked<3>();
    auto Hy_ = Hy.mutable_unchecked<3>();
    auto Hz_ = Hz.mutable_unchecked<3>();
    auto x_range = std::get<0>(update_range);
    auto y_range = std::get<1>(update_range);
    auto z_range = std::get<2>(update_range);
    for(int i = x_range.first; i < x_range.second; i++){
        for(int j = y_range.first; j < y_range.second; j++){
            for(int k = z_range.first; k < z_range.second; k++){
                int materialHx = ID_(3, i, j, k);
                int materialHy = ID_(4, i, j, k);
                int materialHz = ID_(5, i, j, k);
                if(j < ymax - 1 && k < zmax - 1)
                    Hx_(i, j, k) = (updatecoeffsH_(materialHx, 0) * Hx_(i, j, k) -
                                    updatecoeffsH_(materialHx, 2) * (Ez_(i, j + 1, k) - Ez_(i, j, k)) +
                                    updatecoeffsH_(materialHx, 3) * (Ey_(i, j, k + 1) - Ey_(i, j, k)));
                if(i < xmax - 1 && k < zmax - 1)
                    Hy_(i, j, k) = (updatecoeffsH_(materialHy, 0) * Hy_(i, j, k) -
                                    updatecoeffsH_(materialHy, 3) * (Ex_(i, j, k + 1) - Ex_(i, j, k)) +
                                    updatecoeffsH_(materialHy, 1) * (Ez_(i + 1, j, k) - Ez_(i, j, k)));
                if(i < xmax - 1 && j < ymax - 1)
                    Hz_(i, j, k) = (updatecoeffsH_(materialHz, 0) * Hz_(i, j, k) -
                                    updatecoeffsH_(materialHz, 1) * (Ey_(i + 1, j, k) - Ey_(i, j, k)) +
                                    updatecoeffsH_(materialHz, 2) * (Ex_(i, j + 1, k) - Ex_(i, j, k)));
            }
        }
    }
}

    