#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include "vec.h"
#include "constant.h"
#include "H5Cpp.h"

using namespace H5;
using namespace std;


/* Macro Function: load_hdf5_model
 * Usage: load_hdf5_model(model_type, dataset_path, data_type);
 * Parameters:
 * 		model_type:   data name, e.g. shape_mu;
 * 		dataset_path: dataset path in .h5 file, e.g. "/shape/model/mean"
 * 		data_type:    data type in dataset, e.g. PredType::NATIVE_FLOAT
 * ***********************************************************************
 * Load data from .h5 format file into corresponding data struction.
 */

#define load_hdf5_model(model_type, dataset_path, data_type) { \
			DataSet model_type##_data = file.openDataSet(dataset_path); \
			model_type##_data.read(model_type##_raw, data_type); \
			model_type##_data.close(); \
			raw2vector(model_type, model_type##_raw); \
		} 


/* Function: raw2vector
 * Usage: raw2vector(ev, ev_raw);
 * **********************************************************************
 * Convert raw data array to practical vector. 
 * This type is used for ev(shape_ev, or tex_ev).
 */

void raw2vector(std::vector<double> &vec, float *raw);


/* Function: raw2vector
 * Usage: raw2vector(mu, mu_raw);
 * **********************************************************************
 * Convert raw data array to practical vector. 
 * This type is used for mu(shape_mu, or tex_mu).
 */

void raw2vector(std::vector<point3f> &vec, float *raw);


/* Function: raw2vector
 * Usage: raw2vector(tl, tl_raw);
 * **********************************************************************
 * Convert raw data array to practical vector. 
 * This type is used for tl.
 */

void raw2vector(std::vector<point3d> &vec, unsigned int *raw);


/* Function: raw2vector
 * Usage: raw2vector(pc, pc_raw);
 * **********************************************************************
 * Convert raw data array to practical vector. 
 * This type is used for pc(shape_pc, or tex_pc).
 */

void raw2vector(std::vector<std::vector<point3f>> &vec, float *raw);


/* Function: load
 * Usage: load();
 * Caller: bfm.cpp
 * **********************************************************************
 * Load data.
 */

int load();


/* Function: data_check
 * Usage: data_check();
 * Caller: Users
 * **********************************************************************
 * Compare loaded data with hardcoded data to check validity.
 */

void data_check();