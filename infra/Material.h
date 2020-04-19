#pragma once

#include <Ray.h>
#include <Hitable.h>
#include <curand_kernel.h>

struct hit_record; // Defined in Hitable.h

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state);
__device__ vec3 random_in_hemisphere(const vec3& normal,curandState *local_rand_state);
__device__ vec3 reflect(const vec3& v,const vec3& normal);

class material {
	public:
		__device__ virtual bool scatter(const ray& r_in,const hit_record& rec,vec3& attenuation,ray& scattered,curandState * local_rand_state) const = 0;
};

class lambertian : public material {
	public:
		vec3 albedo;
		__device__ lambertian(const vec3&a);
		__device__ bool scatter(const ray& r_in,const hit_record& rec,vec3& attenuation,ray& scattered,curandState * local_rand_state) const;
};