#include <Material.h>

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
	vec3 p;
	do {
		p = 2.0f*RANDVEC3 - vec3(1,1,1);
	} while (p.squared_length() >= 1.0f);
	return p;
}

__device__ vec3 random_in_hemisphere(const vec3& normal,curandState *local_rand_state) {
	vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);
	if (dot(in_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}

__device__ vec3 reflect(const vec3& v,const vec3& n){
	return v - 2.0f*dot(v,n)*n;
}


__device__ lambertian::lambertian(const vec3&a) : albedo(a) {}
__device__ bool lambertian::scatter(const ray&r_in,const hit_record&rec,vec3& attenuation,ray& scattered,curandState *local_rand_state) const{
	vec3 target = rec.p + random_in_hemisphere(rec.normal,local_rand_state);
	scattered = ray(rec.p, target-rec.p);
	attenuation = albedo;
	return true;
}