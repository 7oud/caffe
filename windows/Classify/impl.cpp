#include "Classify.h"
#include "impl_interface.h"

using namespace std;

Classifier* g_classifier = NULL;

int init_model(const string& model_file, 
	const string& trained_file, 
	const string& mean_file, 
	const string& label_file)
{
	//levels INFO, WARNING, ERROR, and FATAL are 0, 1, 2, and 3
	::FLAGS_minloglevel = 4;

	g_classifier = new Classifier(model_file, trained_file, mean_file, label_file);

	int ret = 0;
	if (g_classifier == NULL)
		ret = -1;

	return ret;
}

int prepare_batch()
{
	cout << "prepare_batch" << endl;
	return 0;
}

int predict_batch()
{
	cout << "predict_batch" << endl;
	return 0;
}

int release_model()
{
	if (g_classifier != NULL)
	{
		delete g_classifier;
		g_classifier = NULL;
	}

	return 0;
}