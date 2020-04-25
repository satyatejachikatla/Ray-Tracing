#pragma once

#include <Ray.h>

class camera {
	public:
		__device__ camera(vec3 lookfrom,vec3 lookat,vec3 up,float vfov, float aspect);
		__device__ ray get_ray(float u, float v);

		vec3 origin;
		vec3 lower_left_corner;
		vec3 horizontal;
		vec3 vertical;
};