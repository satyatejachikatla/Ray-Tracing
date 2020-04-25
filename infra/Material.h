#pragma once

#include <Ray.h>
#include <Hitable.h>
#include <curand_kernel.h>

struct hit_record; // Defined in Hitable.h

class material {
	public:
		__device__ virtual bool scatter(const ray& r_in,const hit_record& rec,vec3& attenuation,ray& scattered,curandState * local_rand_state) const = 0;
};

class lambertian : public material {
	public:
		vec3 albedo;
		__device__ lambertian(const vec3& a);
		__device__ bool scatter(const ray& r_in,const hit_record& rec,vec3& attenuation,ray& scattered,curandState * local_rand_state) const;
};

class metal : public material {
	public:
		vec3 albedo;
		float fuzz;
		__device__ metal(const vec3& a,float f);
		__device__ bool scatter(const ray& r_in,const hit_record& rec,vec3& attenuation,ray& scattered,curandState * local_rand_state) const;
};

class dielectric : public material {
	public:
		float ref_idx;
		__device__ dielectric(float ri);
		__device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered,curandState * local_rand_state) const;
};