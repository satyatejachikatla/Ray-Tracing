//  Tutorial from : https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/

#include <iostream>
#include <time.h>

#include <curand_kernel.h>

#include <cudaErrors.h>

#include <Render.h>
#include <Vector.h>
#include <Ray.h>
#include <Camera.h>

#include <ImageHelper.h>

int main() {

	//Adding Clock to profile//
	clock_t start,stop;

	start = clock();

	// Window size //
	const unsigned int nx = 1200;
	const unsigned int ny = 600;
	unsigned int ns = 100; // Number of samples per pixel for anti aliasing
	unsigned int num_pixels = nx*ny;

	// Host/GPU Device mem allocation //
	// vec3 because of RGB //
	size_t fb_size = num_pixels*sizeof(vec3);
	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	// The image is broken in to tx x ty shape//
	// GPU Thread block dimention subject to change //
	unsigned int tx = 8;
	unsigned int ty = 8;
	dim3 blocks(nx/tx+1,ny/ty+1);
	dim3 threads(tx,ty);

	//Curand state var for each pixel//
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

	// Init the curand Variables //
	render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Objects init on GPU //
	hitable **d_list;
	checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hitable *))); //2 objects will be created in create world
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

	create_world<<<1,1>>>(d_list,d_world,d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Call to renderer //
	render<<<blocks,threads>>>(fb,nx,ny,ns,
								d_camera,                  // Camera
								d_world,                   // Za warudo
								d_rand_state               // Curand state per pixel
								);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();

	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Time per frame : "<<timer_seconds << " seconds"  << std::endl;

	// Output to img //
	saveImage(fb,nx,ny,"Img.jpg");

	// Clean up //
	free_world<<<1,1>>>(d_list,d_world,d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();

	return 0;
}