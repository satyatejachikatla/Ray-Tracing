#include <cfloat>

#include <Render.h>
#include <Hitable.h>
#include <Objects/Sphere.h>


__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j*max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*(d_list)   = new sphere(vec3(0,0,-1), 0.5);
		*(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
		*d_world    = new hitable_list(d_list,2);
		*d_camera   = new camera();
	}
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	delete *(d_list);
	delete *(d_list+1);
	delete *d_world;
	delete *d_camera;
}

__device__ vec3 color(const ray& r, hitable **world) {
	hit_record rec;
	if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
		return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
	}
	else {
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5f*(unit_direction.y() + 1.0f);
		return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
	}
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j*max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0,0,0);
	for(int s=0; s < ns; s++) {
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u,v);
		col += color(r, world);
	}
	fb[pixel_index] = col/float(ns);
}

