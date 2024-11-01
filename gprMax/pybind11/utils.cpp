#include "inc/utils.h"

std::pair<int, int> GetRange(const std::string& shape, const std::string& electric_or_magnetic, int time_block_size, int space_block_size, int sub_timestep, int tile_index, int start, int end) {
    auto between = [](int first, int second, int min, int max) {
        auto clamp = [](int val, int min, int max) {
            return std::max(min, std::min(val, max));
        };
        first = clamp(first, min, max);
        second = clamp(second, min, max);
        return std::make_pair(first, second);
    };

    int start_idx, end_idx;

    if (shape == "m") {
        if (electric_or_magnetic == "E") {
            if (tile_index == 0) {
                start_idx = start;
                end_idx = start_idx + time_block_size + space_block_size - sub_timestep;
                return between(start_idx, end_idx, start, end);
            } else {
                start_idx = start + 2 * space_block_size + time_block_size + sub_timestep + (tile_index - 1) * (2 * space_block_size + 2 * time_block_size - 1);
                end_idx = start_idx + space_block_size + 2 * time_block_size - 1 - 2 * sub_timestep;
                return between(start_idx, end_idx, start, end);
            }
        } else if (electric_or_magnetic == "H") {
            if (tile_index == 0) {
                start_idx = start;
                end_idx = start_idx + time_block_size + space_block_size - 1 - sub_timestep;
                return between(start_idx, end_idx, start, end);
            } else {
                start_idx = start + 2 * space_block_size + time_block_size + sub_timestep + (tile_index - 1) * (2 * space_block_size + 2 * time_block_size - 1);
                end_idx = start_idx + space_block_size + 2 * time_block_size - 2 - 2 * sub_timestep;
                return between(start_idx, end_idx, start, end);
            }
        }
    } else if (shape == "v") {
        if (electric_or_magnetic == "E") {
            if (tile_index == 0) {
                start_idx = start + time_block_size + space_block_size - sub_timestep;
                end_idx = start_idx + space_block_size + 2 * sub_timestep;
                return between(start_idx, end_idx, start, end);
            } else {
                start_idx = start + 2 * space_block_size + time_block_size + sub_timestep + (tile_index - 1) * (2 * space_block_size + 2 * time_block_size - 1);
                start_idx = start_idx + space_block_size + 2 * time_block_size - 1 - 2 * sub_timestep;
                end_idx = start_idx + space_block_size + 2 * sub_timestep;
                return between(start_idx, end_idx, start, end);
            }
        } else if (electric_or_magnetic == "H") {
            if (tile_index == 0) {
                start_idx = start + time_block_size + space_block_size - 1 - sub_timestep;
                end_idx = start_idx + space_block_size + 1 + 2 * sub_timestep;
                return between(start_idx, end_idx, start, end);
            } else {
                start_idx = start + 2 * space_block_size + time_block_size + sub_timestep + (tile_index - 1) * (2 * space_block_size + 2 * time_block_size - 1);
                start_idx = start_idx + space_block_size + 2 * time_block_size - 2 - 2 * sub_timestep;
                end_idx = start_idx + space_block_size + 1 + 2 * sub_timestep;
                return between(start_idx, end_idx, start, end);
            }
        }
    } else if (shape == "p") {
        if (electric_or_magnetic == "E") {
            start_idx = start + tile_index * space_block_size - sub_timestep;
            end_idx = start_idx + space_block_size;
            return between(start_idx, end_idx, start, end);
        } else if (electric_or_magnetic == "H") {
            start_idx = start + tile_index * space_block_size - sub_timestep - 1;
            end_idx = start_idx + space_block_size;
            return between(start_idx, end_idx, start, end);
        }
    }

    // Default return if none of the conditions match
    return std::make_pair(start, end);
}