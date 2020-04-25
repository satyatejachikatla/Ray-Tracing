#include<Camera.h>

__device__ camera::camera(vec3 lookfrom,vec3 lookat,vec3 vup,float vfov, float aspect) {
	//vfov in degress

	origin = lookfrom;

	vec3 u,v,w;
	float theta = vfov*M_PI/180;
	float half_height= tan(theta/2);
	float half_width = aspect * half_height;

	w = unit_vector(lookfrom-lookat);
	u = unit_vector(cross(vup,w));
	v = cross(w,u);

	lower_left_corner = origin - half_width*u - half_height*v -w ;

	horizontal = 2*half_width*u;
	vertical   = 2*half_height*v;

}
__device__ ray camera::get_ray(float u, float v) { 
	return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
	//return ray(lower_left_corner + u*horizontal + v*vertical - vec3(0,0,-1),vec3(0.0f,0.0f,-1.0f));
}
