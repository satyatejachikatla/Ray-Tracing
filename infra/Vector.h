#pragma once

//Code coppied from : https://github.com/rogerallen/raytracinginoneweekendincuda/blob/ch02_vec3_cuda/vec3.h

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3  {


public:
	__host__ __device__ vec3() {}
	__host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }

	__host__ __device__ inline float x() const { return e[0]; }
	__host__ __device__ inline float y() const { return e[1]; }
	__host__ __device__ inline float z() const { return e[2]; }

	__host__ __device__ inline float r() const { return e[0]; }
	__host__ __device__ inline float g() const { return e[1]; }
	__host__ __device__ inline float b() const { return e[2]; }

	__host__ __device__ inline const vec3& operator+() const { return *this; }
	__host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

	__host__ __device__ inline float operator[](int i) const { return e[i]; }
	__host__ __device__ inline float& operator[](int i) { return e[i]; };

	__host__ __device__ inline vec3& operator+=(const vec3 &v2);
	__host__ __device__ inline vec3& operator-=(const vec3 &v2);
	__host__ __device__ inline vec3& operator*=(const vec3 &v2);
	__host__ __device__ inline vec3& operator/=(const vec3 &v2);
	__host__ __device__ inline vec3& operator*=(const float t);
	__host__ __device__ inline vec3& operator/=(const float t);


    __host__ __device__ inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    __host__ __device__ inline float squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    __host__ __device__ inline void make_unit_vector();

    float e[3];

};

inline std::istream& operator>>(std::istream &is, vec3 &t);
inline std::ostream& operator<<(std::ostream &os, const vec3 &t);
__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2);
__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2);
__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2);
__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2);
__host__ __device__ inline vec3 operator*(float t, const vec3 &v);
__host__ __device__ inline vec3 operator/(vec3 v, float t);
__host__ __device__ inline vec3 operator*(const vec3 &v, float t) ;
__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2);
__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2);
__host__ __device__ inline vec3 unit_vector(vec3 v);

