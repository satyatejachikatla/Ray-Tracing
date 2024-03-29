#include <ImageHelper.h>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vector>

void saveImagePPM(float* fb,unsigned int nx,unsigned int ny,const char* filename){

	std::ofstream file(filename,std::ofstream::out);
	file << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny-1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j*3*nx + i*3;
			float r = fb[pixel_index + 0];
			float g = fb[pixel_index + 1];
			float b = fb[pixel_index + 2];
			int ir = int(255.99*r);
			int ig = int(255.99*g);
			int ib = int(255.99*b);
			file << ir << " " << ig << " " << ib << "\n";
		}
	}
	file.close();
}

void saveImagePPM(vec3* fb,unsigned int nx,unsigned int ny,const char* filename){

	std::ofstream file(filename,std::ofstream::out);
	file << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny-1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j*nx + i;
			int ir = int(255.99*fb[pixel_index].r());
			int ig = int(255.99*fb[pixel_index].g());
			int ib = int(255.99*fb[pixel_index].b());
			file << ir << " " << ig << " " << ib << "\n";
		}
	}
	file.close();
}

void saveImage(vec3* fb,unsigned int nx,unsigned int ny,const char* filename){

	cv::Mat img(ny,nx,CV_8UC3,cv::Scalar(0,0,0));

	for (int j = ny-1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j*nx + i;
			int ir = int(255.99*fb[pixel_index].r());
			int ig = int(255.99*fb[pixel_index].g());
			int ib = int(255.99*fb[pixel_index].b());

			img.at<cv::Vec3b>(ny-j-1,i)[0] = ib;/*B*/
			img.at<cv::Vec3b>(ny-j-1,i)[1] = ig;/*G*/
			img.at<cv::Vec3b>(ny-j-1,i)[2] = ir;/*R*/
		}
	}

	cv::imwrite(filename,img);

}