//  Tutorial from : https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/

#include <iostream>
#include <cudaErrors.h>

#include <Render.h>
#include <Vector.h>
#include <Ray.h>

#include <ImageHelper.h>

int main() {

	// Window size //
	const unsigned int nx = 640;
	const unsigned int ny = 480;
	unsigned int num_pixels = nx*ny;

	// Host/GPU Device mem allocation //
	// 3 because of RGB //
	size_t fb_size = 3*num_pixels*sizeof(float);
	float *fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	// The image is broken in to tx x ty shape//
	// GPU Thread block dimention subject to change //
	unsigned int tx = 8;
	unsigned int ty = 8;
	dim3 blocks(nx/tx+1,ny/ty+1);
	dim3 threads(tx,ty);

	// Call to renderer //
	render<<<blocks,threads>>>(fb,nx,ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Output to PPM //
	saveImagePPM(fb,nx,ny,"Img.ppm");

	// Clean up //
	checkCudaErrors(cudaFree(fb));

	return 0;
}