#include <cfloat>

#include <Render.h>
#include <Hitable.h>
#include <Material.h>
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
		*(d_list)   = new sphere(vec3(0,0,-1), 0.5,
									new lambertian(vec3(0.8,0.3,0.3)));
		*(d_list+1) = new sphere(vec3(0,-100.5,-1), 100,
									new lambertian(vec3(0.8,0.8,0.0)));
		*d_world    = new hitable_list(d_list,2);
		*d_camera   = new camera();
	}
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	delete ((sphere*)d_list[0])->mat_ptr;
	delete ((sphere*)d_list[1])->mat_ptr;
	delete  d_list[0];
	delete  d_list[1];
	delete *d_world;
	delete *d_camera;
}

__device__ vec3 color(const ray& r, hitable **world,curandState *local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation(1.0f,1.0f,1.0f);
	for(int i = 0; i < 50; i++) { // Here 50 bounces of ray is max
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if(rec.mat_ptr->scatter(cur_ray,rec,attenuation,scattered,local_rand_state)){
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0.0f,0.0f,0.0f);
			}
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f*(unit_direction.y() + 1.0f);
			vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0,0.0,0.0); // exceeded recursion
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
		col += color(r, world,&local_rand_state);
	}
	fb[pixel_index] = col/float(ns);
}

