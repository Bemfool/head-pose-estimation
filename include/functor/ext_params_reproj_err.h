#ifndef HPE_EXT_PARAMS_REPROJ_ERR_H
#define HPE_EXT_PARAMS_REPROJ_ERR_H

#include "bfm_manager.h"
#include "ceres/ceres.h"
#include "db_params.h"
#include "io_utils.h"

using Eigen::VectorXd;
using Eigen::Matrix;
using Eigen::Dynamic;

class ExtParamsReprojErr 
{
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ExtParamsReprojErr(dlib::full_object_detection *observedPoints, BaselFaceModelManager *model, std::vector<unsigned int> &aLandmarkMap) 
	: m_pObservedPoints(observedPoints), m_pModel(model), m_aLandmarkMap(aLandmarkMap) { }
	
    template<typename _Tp>
	bool operator () (const _Tp* const aExtParams, _Tp* aResiduals) const 
	{
		_Tp fx = _Tp(m_pModel->getFx()), fy = _Tp(m_pModel->getFy());
		_Tp cx = _Tp(m_pModel->getCx()), cy = _Tp(m_pModel->getCy());
		const VectorXd vecLandmarkBlendshape = m_pModel->getLandmarkCurrentBlendshape();
		const Matrix<_Tp, Dynamic, 1> vecLandmarkBlendshapeTransformed = bfm_utils::TransPoints(aExtParams, vecLandmarkBlendshape);

		for(int iLandmark = 0; iLandmark < N_LANDMARK; iLandmark++) 
		{
			int iDlibLandmarkIdx = m_aLandmarkMap[iLandmark];
			_Tp u = fx * vecLandmarkBlendshapeTransformed(iLandmark * 3) / vecLandmarkBlendshapeTransformed(iLandmark * 3 + 2) + cx;
			_Tp v = fy * vecLandmarkBlendshapeTransformed(iLandmark * 3 + 1) / vecLandmarkBlendshapeTransformed(iLandmark * 3 + 2) + cy;
			aResiduals[iLandmark * 2] = _Tp(m_pObservedPoints->part(iDlibLandmarkIdx).x()) - u;
			aResiduals[iLandmark * 2 + 1] = _Tp(m_pObservedPoints->part(iDlibLandmarkIdx).y()) - v;
		}
		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection *observedPoints, BaselFaceModelManager *model, std::vector<unsigned int> aLandmarkMap) 
	{
		return (new ceres::AutoDiffCostFunction<ExtParamsReprojErr, N_LANDMARK * 2, N_EXT_PARAMS>(
			new ExtParamsReprojErr(observedPoints, model, aLandmarkMap)));
	}

private:
	dlib::full_object_detection *m_pObservedPoints;
    BaselFaceModelManager *m_pModel;
	std::vector<unsigned int> m_aLandmarkMap;
};


class LinearizedExtParamsReprojErr 
{
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	LinearizedExtParamsReprojErr(dlib::full_object_detection *observedPoints, BaselFaceModelManager *model, std::vector<unsigned int> aLandmarkMap) 
	: m_pObservedPoints(observedPoints), m_pModel(model), m_aLandmarkMap(aLandmarkMap) { }

	LinearizedExtParamsReprojErr(dlib::full_object_detection *observedPoints, BaselFaceModelManager *model, std::vector<unsigned int> aLandmarkMap, double a, double b) 
	: m_pObservedPoints(observedPoints), m_pModel(model), m_aLandmarkMap(aLandmarkMap), m_dRotWeight(a), m_dTranWeight(b) {}


    template<typename _Tp>
	bool operator () (const _Tp* const aExtParams, _Tp* aResiduals) const 
	{
		_Tp fx = _Tp(m_pModel->getFx()), fy = _Tp(m_pModel->getFy());
		_Tp cx = _Tp(m_pModel->getCx()), cy = _Tp(m_pModel->getCy());

		const VectorXd vecLandmarkBlendshape = m_pModel->getLandmarkCurrentBlendshapeTransformed();
		const Matrix<_Tp, Dynamic, 1> vecLandmarkBlendshapeTransformed = bfm_utils::TransPoints(aExtParams, vecLandmarkBlendshape, true);

		for(int iLandmark = 0; iLandmark < N_LANDMARK; iLandmark++) 
		{
			int iDlibLandmarkIdx = m_aLandmarkMap[iLandmark];
			_Tp u = fx * vecLandmarkBlendshapeTransformed(iLandmark * 3) / vecLandmarkBlendshapeTransformed(iLandmark * 3 + 2) + cx;
			_Tp v = fy * vecLandmarkBlendshapeTransformed(iLandmark * 3 + 1) / vecLandmarkBlendshapeTransformed(iLandmark * 3 + 2) + cy;
			aResiduals[iLandmark * 2] = _Tp(m_pObservedPoints->part(iDlibLandmarkIdx).x()) - u;
			aResiduals[iLandmark * 2 + 1] = _Tp(m_pObservedPoints->part(iDlibLandmarkIdx).y()) - v;
		}

		/* regularialization */
	    aResiduals[N_LANDMARK * 2]   = _Tp(m_dRotWeight) * aExtParams[0];
		aResiduals[N_LANDMARK * 2 + 1] = _Tp(m_dRotWeight) * aExtParams[1];
		aResiduals[N_LANDMARK * 2 + 2] = _Tp(m_dRotWeight) * aExtParams[2];
		aResiduals[N_LANDMARK * 2 + 3] = _Tp(m_dTranWeight) * aExtParams[3];
		aResiduals[N_LANDMARK * 2 + 4] = _Tp(m_dTranWeight) * aExtParams[4];
		aResiduals[N_LANDMARK * 2 + 5] = _Tp(m_dTranWeight) * aExtParams[5];
		// print_array(aResiduals+N_LANDMARK*2, 6);
		
		return true;
	}

	static ceres::CostFunction *create(dlib::full_object_detection *observedPoints, BaselFaceModelManager *model, std::vector<unsigned int> aLandmarkMap) 
	{
		return (new ceres::AutoDiffCostFunction<LinearizedExtParamsReprojErr, N_LANDMARK * 2 + N_EXT_PARAMS, N_EXT_PARAMS>(
			new LinearizedExtParamsReprojErr(observedPoints, model, aLandmarkMap)));
	}

	static ceres::CostFunction *create(dlib::full_object_detection *observedPoints, BaselFaceModelManager *model, std::vector<unsigned int> aLandmarkMap, double a, double b) 
	{
		return (new ceres::AutoDiffCostFunction<LinearizedExtParamsReprojErr, N_LANDMARK * 2 + N_EXT_PARAMS, N_EXT_PARAMS>(
			new LinearizedExtParamsReprojErr(observedPoints, model, aLandmarkMap, a, b)));
	}


private:
	dlib::full_object_detection *m_pObservedPoints;
    BaselFaceModelManager *m_pModel;
	std::vector<unsigned int> m_aLandmarkMap;
	/* residual coefficients */
	double m_dRotWeight = 1.0;	/* for rotation */
	double m_dTranWeight = 1.0;	/* for translation */
};


#endif // HPE_EXT_PARAMS_REPROJ_ERR_H