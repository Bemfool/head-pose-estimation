#ifndef HPE_SHAPE_COEF_REPROJ_ERR_H
#define HPE_SHAPE_COEF_REPROJ_ERR_H

#include "bfm_manager.h"
#include "ceres/ceres.h"
#include "db_params.h"

using FullObjectDetection = dlib::full_object_detection;
using Eigen::Matrix;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

class ShapeCoefReprojErr {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ShapeCoefReprojErr(FullObjectDetection *observedPoints, BaselFaceModelManager *model, std::vector<unsigned int> aLandmarkMap) 
		: m_pObservedPoints(observedPoints), m_pModel(model), m_aLandmarkMap(aLandmarkMap) { }
	
    template<typename _Tp>
	bool operator () (const _Tp* const aShapeCoef, _Tp* aResiduals) const {
		_Tp fx = _Tp(m_pModel->getFx()), fy = _Tp(m_pModel->getFy());
		_Tp cx = _Tp(m_pModel->getCx()), cy = _Tp(m_pModel->getCy());
		
		const Matrix<_Tp, Dynamic, 1> vecLandmarkBlendshape = m_pModel->genLandmarkBlendshapeByShape(aShapeCoef);  

		const double *daExtParams = m_pModel->getExtParams();
        _Tp *taExtParams = new _Tp[N_EXT_PARAMS];
        for(unsigned int iParam = 0; iParam < N_EXT_PARAMS; iParam++)
            taExtParams[iParam] = (_Tp)(daExtParams[iParam]);

		const Matrix<_Tp, Dynamic, 1> vecLandmarkBlendshapeTransformed = bfm_utils::TransPoints(taExtParams, vecLandmarkBlendshape);

		for(int iLandmark = 0; iLandmark < N_LANDMARKS; iLandmark++) 
		{
			unsigned int iDlibLandmarkIdx = m_aLandmarkMap[iLandmark];
			_Tp u = fx * vecLandmarkBlendshapeTransformed(iLandmark * 3) / vecLandmarkBlendshapeTransformed(iLandmark * 3 + 2) + cx;
			_Tp v = fy * vecLandmarkBlendshapeTransformed(iLandmark * 3 + 1) / vecLandmarkBlendshapeTransformed(iLandmark * 3 + 2) + cy;
			aResiduals[iLandmark * 2] = _Tp(m_pObservedPoints->part(iDlibLandmarkIdx).x()) - u;
			aResiduals[iLandmark * 2 + 1] = _Tp(m_pObservedPoints->part(iDlibLandmarkIdx).y()) - v;
		}

		return true;
	}

	static CostFunction *create(FullObjectDetection *observedPoints, BaselFaceModelManager *model, std::vector<unsigned int> aLandmarkMap) {
		return (new AutoDiffCostFunction<ShapeCoefReprojErr, N_LANDMARKS * 2, N_ID_PCS>(
			new ShapeCoefReprojErr(observedPoints, model, aLandmarkMap)));
	}

private:
	FullObjectDetection *m_pObservedPoints;
	BaselFaceModelManager *m_pModel;
	std::vector<unsigned int> m_aLandmarkMap;
};


#endif // HPE_SHAPE_COEF_REPROJ_ERR_H