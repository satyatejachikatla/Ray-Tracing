#pragma once

#include <Vector.h>
#include <Ray.h>

__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin);