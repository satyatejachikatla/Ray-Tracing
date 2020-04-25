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

__device__ vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    auto cos_theta = dot(-uv, n);
    vec3 r_out_parallel =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_perp = -sqrt(1.0 - r_out_parallel.squared_length()) * n;
    return r_out_parallel + r_out_perp;
}
__device__ float schlick(float cosine, float ref_idx) {
    auto r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}

__device__ lambertian::lambertian(const vec3& a) : albedo(a) {}
__device__ bool lambertian::scatter(const ray&r_in,const hit_record&rec,vec3& attenuation,ray& scattered,curandState *local_rand_state) const{
	vec3 target = rec.p + random_in_hemisphere(rec.normal,local_rand_state);
	scattered = ray(rec.p, target-rec.p);
	attenuation = albedo;
	return true;
}

__device__ metal::metal(const vec3& a,float f) : albedo(a), fuzz(f) {}
__device__ bool metal::scatter(const ray&r_in,const hit_record&rec,vec3& attenuation,ray& scattered,curandState *local_rand_state) const{
	vec3 reflected = reflect(unit_vector(r_in.direction()),rec.normal);
	scattered = ray(rec.p, reflected + fuzz*random_in_hemisphere(rec.normal,local_rand_state));
	attenuation = albedo;
	return (dot(scattered.direction(),rec.normal) > 0.0f); // We dont want rays coming from inside of the sphere by any chance.
}

__device__ dielectric::dielectric(float ri) : ref_idx(ri) {}
__device__ bool dielectric::scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState * local_rand_state) const {
	attenuation = vec3(1.0f, 1.0f, 1.0f);
	float etai_over_etat;
	if (rec.front_face) {
		etai_over_etat = 1.0f / ref_idx;
	} else {
		etai_over_etat = ref_idx;
	}

	vec3 unit_direction = unit_vector(r_in.direction());

	float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
	float sin_theta = sqrt(1.0f - cos_theta*cos_theta);

	if (etai_over_etat * sin_theta > 1.0f ) {
		vec3 reflected = reflect(unit_direction, rec.normal);
		scattered = ray(rec.p, reflected);
		return true;
	}

	float reflect_prob = schlick(cos_theta, etai_over_etat);
	if (curand_uniform(local_rand_state) < reflect_prob)
	{
		vec3 reflected = reflect(unit_direction, rec.normal);
		scattered = ray(rec.p, reflected);
		return true;
	}

	vec3 refracted = refract(unit_direction, rec.normal, etai_over_etat);
	scattered = ray(rec.p, refracted);
	return true;
}
