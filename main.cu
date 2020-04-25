//  Tutorial from : https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/

#include <iostream>
#include <time.h>
#include <sstream>
#include <iomanip>

#include <curand_kernel.h>

#include <cudaErrors.h>

#include <Render.h>
#include <Vector.h>
#include <Ray.h>
#include <Camera.h>

#include <ImageHelper.h>

int main() {

	// Window size //
	const unsigned int nx = 1200;
	const unsigned int ny = 600;
	unsigned int ns = 200; // Number of samples per pixel for anti aliasing
	unsigned int num_pixels = nx*ny;

	// Host/GPU Device mem allocation //
	// vec3 because of RGB //
	size_t fb_size = num_pixels*sizeof(vec3);
	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	// The image is broken in to tx x ty shape//
	// GPU Thread block dimention subject to change //
	unsigned int tx = 32;
	unsigned int ty = 32;
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

	int num_hitables = 22*22+1+3;

	hitable **d_list;
	checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

	create_world<<<1,1>>>(d_list,d_world,d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	unsigned int totalsteps = 100;

	for(unsigned int step = 0 ;step < totalsteps ; step++) {
		//Adding Clock to profile//
		clock_t start,stop;

		start = clock();

		// Init Camera //

		init_cam<<<1,1>>>(d_camera,nx,ny,step,totalsteps);
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


		// Delete Cam //

		delete_cam<<<1,1>>>(d_camera);
		checkCudaErrors(cudaGetLastError());

		stop = clock();

		double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
		std::cout << "Time per frame : "<<timer_seconds << " seconds"  << std::endl;

		std::stringstream ss;
		ss << "./save_jpgs/Img-" << std::setfill('0') << std::setw(5) << step << ".jpg";

		// Output to img //
		saveImage(fb,nx,ny,ss.str().c_str());
	}

	// Clean up //
	free_world<<<1,1>>>(d_list,d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();

	return 0;
}