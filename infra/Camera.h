#pragma once

#include <Ray.h>
#include <curand_kernel.h>

class camera {
	public:
		__device__ camera(vec3 lookfrom,vec3 lookat,vec3 vup,float vfov, float aspect,float aperture,float focus_dist) ;
		__device__ ray get_ray(float u, float v, curandState *local_rand_state);

		vec3 u,v,w;
		vec3 origin;
		vec3 lower_left_corner;
		vec3 horizontal;
		vec3 vertical;
		float lens_radius;
};