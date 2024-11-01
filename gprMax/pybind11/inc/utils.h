#ifndef UTILS_H
#define UTILS_H

#include "common.h"

std::pair<int, int> GetRange(
    const std::string& shape, 
    const std::string& electric_or_magnetic, 
    int time_block_size, 
    int space_block_size, 
    int sub_timestep, 
    int tile_index, 
    int start, 
    int end
);

#endif // UTILS_H
