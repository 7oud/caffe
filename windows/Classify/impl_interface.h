
#ifndef __IMPL_INTERFACE_H__
#define __IMPL_INTERFACE_H__

#include <iostream>

using std::string;

extern "C"
{
	__declspec(dllexport) int init_model(const string& model_file, const string& trained_file,
		const string& mean_file, const string& label_file);
	__declspec(dllexport) int prepare_batch();
	__declspec(dllexport) int predict_batch();
	__declspec(dllexport) int release_model();

}


#endif
