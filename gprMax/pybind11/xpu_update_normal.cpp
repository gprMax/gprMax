#include "inc/xpu_update.h"

void xpu_update::update_electric_normal(update_range_t update_range){
    std::cout << "update_range_E: " 
    << std::get<0>(update_range).first << ", " << std::get<0>(update_range).second << ", " 
    << std::get<1>(update_range).first << ", " << std::get<1>(update_range).second << ", " 
    << std::get<2>(update_range).first << ", " << std::get<2>(update_range).second << std::endl;
}

void xpu_update::update_magnetic_normal(update_range_t update_range){
    std::cout << "update_range_H: " 
    << std::get<0>(update_range).first << ", " << std::get<0>(update_range).second << ", " 
    << std::get<1>(update_range).first << ", " << std::get<1>(update_range).second << ", " 
    << std::get<2>(update_range).first << ", " << std::get<2>(update_range).second << std::endl;
}

    