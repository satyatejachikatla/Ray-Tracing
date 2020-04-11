//  Tutorial from : https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/

#include <iostream>
#include <time.h>

#include <cudaErrors.h>

#include <Render.h>
#include <Vector.h>
#include <Ray.h>

#include <ImageHelper.h>

int main() {

	//Adding Clock to profile//
	clock_t start,stop;

	// Window size //
	const unsigned int nx = 1200;
	const unsigned int ny = 600;
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

	start = clock();

	//Objects init on GPU
	hitable **d_list;
	checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hitable *))); //2 will be created in create world
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
	create_world<<<1,1>>>(d_list,d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Call to renderer //
	render<<<blocks,threads>>>(fb,nx,ny,
								vec3(-2.0f, -1.0f, -1.0f), // Left Bottom corner
								vec3( 4.0f,  0.0f,  0.0f), // X - axis width from Left Boot corner
								vec3( 0.0f,  2.0f,  0.0f), // Y - axis height from Left Boot corner
								vec3( 0.0f,  0.0f,  0.0f), // Origin Location.
								d_world                    // Za warudo
								);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();

	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Time per frame : "<<timer_seconds << " seconds"  << std::endl;

	// Output to PPM //
	saveImage(fb,nx,ny,"Img.jpg");

	// Clean up //
	checkCudaErrors(cudaFree(fb));

	return 0;
}