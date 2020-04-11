#pragma once

#include <Vector.h>

void saveImagePPM(float* fb,unsigned int nx,unsigned int ny,const char* filename);
void saveImagePPM(vec3* fb,unsigned int nx,unsigned int ny,const char* filename);

void saveImage(vec3* fb,unsigned int nx,unsigned int ny,const char* filename);