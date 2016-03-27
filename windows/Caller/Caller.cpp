// Caller.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>

#include <opencv2/core/core.hpp>

#include "../Classify/impl_interface.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	int ret = -1;

	string model_file = "..\\..\\models\\bvlc_reference_caffenet\\deploy.prototxt";
	string trained_file = "..\\..\\models\\bvlc_reference_caffenet\\bvlc_reference_caffenet.caffemodel";
	string mean_file = "..\\..\\data\\ilsvrc12\\imagenet_mean.binaryproto";
	string label_file = "..\\..\\data\\ilsvrc12\\synsets.txt";

	if (0 != init_model(model_file, trained_file, mean_file, label_file))
	{
		cerr << "init_model error!" << endl;
	}

	if (0 != prepare_batch())
	{
		cerr << "prepare_batch error!" << endl;
	}

	if (0 != predict_batch())
	{
		cerr << "predict_batch error!" << endl;
	}

	if (0 != release_model())
	{
		cerr << "release_model error!" << endl;
	}

	return 0;
}

