#pragma once

//https://github.com/rogerallen/raytracinginoneweekendincuda/tree/ch05_normals_cuda

#include <Ray.h>
#include <Material.h>

class material; // Defined in Material.h

struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;

	bool front_face;
	material* mat_ptr;

	__device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hitable  {
	public:
		__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

class hitable_list: public hitable  {
	public:
		__device__ hitable_list();
		__device__ hitable_list(hitable **l, int n);
		__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
		hitable **list;
		int list_size;
};