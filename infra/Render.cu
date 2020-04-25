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

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera,unsigned int nx,unsigned int ny,curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;

		*d_world = new hitable_list(d_list, 22*22+1+3);

		d_list[0]    = new sphere(vec3(0,-1000.0f,-1), 1000,
									new lambertian(vec3(float(0x87), float(0xd3), float(0x7c))/vec3(256.0f,256.0f,256.0f)));

		int i = 1;
		
		for(int a = -11 ; a < 11 ; a++) {
			for(int b = -11 ; b < 11 ; b++ ) {
				float chose_mat = RND;
				vec3 center (a+RND,0.2,b+RND);
				if(chose_mat < 0.8f) {
					d_list[i] = new sphere(center, 0.2,
											 new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
				} else if(chose_mat < 0.95f) {
					d_list[i] = new sphere(center, 0.2,
											 new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
				} else {
					d_list[i] = new sphere(center, 0.2, new dielectric(1.5));
				}
				i+=1;
			}
		}

		d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
		d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
		d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

		vec3 lookfrom = vec3(13,2,3);
		vec3 lookat   = vec3(0,0,0);

		float dist_to_focus = (lookfrom-lookat).length();
		float aperture = 0.1f;

		
		*d_camera   = new camera(lookfrom,
								 lookat,
								 vec3(0,1,0),
								 30.0f,
								 float(nx)/float(ny),
								 aperture,
								 dist_to_focus);
	}
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	for(int i=0; i<22*22+3+1; i++) {
		delete ((sphere*)d_list[i])->mat_ptr;
		delete  d_list[i];
	}
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
			vec3 c = (1.0f-t)*vec3(1.0f, 1.0f, 1.0f) + t*vec3(0.5f, 0.7f, 1.0f);
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
		ray r = (*cam)->get_ray(u,v,&local_rand_state);
		col += color(r, world,&local_rand_state);
	}
	fb[pixel_index] = col/float(ns);
}

