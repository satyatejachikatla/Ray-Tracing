#pragma once

#include <Hitable.h>
#include <Vector.h>
#include <Ray.h>
#include <Camera.h>
#include <curand_kernel.h>

__global__ void render_init(int max_x, int max_y, curandState *rand_state);
__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state);

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera);
__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera);

