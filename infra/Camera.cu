#include<Camera.h>

__device__ camera::camera() {
	lower_left_corner = vec3(-2.0f, -1.0f, -1.0f); // Left Bottom corner
	horizontal        = vec3( 4.0f,  0.0f,  0.0f); // X - axis width from Left Boot corner
	vertical          = vec3( 0.0f,  2.0f,  0.0f); // Y - axis height from Left Boot corner
	origin            = vec3( 0.0f,  0.0f,  0.0f); // Origin Location.

}
__device__ ray camera::get_ray(float u, float v) { 
	return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
	//return ray(lower_left_corner + u*horizontal + v*vertical - vec3(0,0,-1),vec3(0.0f,0.0f,-1.0f));
}
