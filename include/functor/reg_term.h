#ifndef HPE_REG_TERM_H
#define HPE_REG_TERM_H

#include "bfm_manager.h"
#include "ceres/ceres.h"
#include "db_params.h"


class ShapeCoefRegTerm {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ShapeCoefRegTerm(BaselFaceModelManager *pModel) : m_pModel(pModel) {}
    template<typename _Tp>
	bool operator () (_Tp const* const* aParams, _Tp* aResiduals) const {
		const _Tp* aShapeCoefs = aParams[0]; 
        const unsigned int nIdPcs = m_pModel->getNIdPcs();
		for(unsigned int iCoef = 0; iCoef < nIdPcs; iCoef++)
	        aResiduals[iCoef] = _Tp(m_dWeight) * aShapeCoefs[iCoef];
		return true;
	}

	static ceres::DynamicAutoDiffCostFunction<ShapeCoefRegTerm> *create(BaselFaceModelManager *pModel) {
		return (new ceres::DynamicAutoDiffCostFunction<ShapeCoefRegTerm>(new ShapeCoefRegTerm(pModel)));
	}

private:
	const double m_dWeight = 0.003;
	BaselFaceModelManager *m_pModel;
};


class ExprCoefRegTerm {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ExprCoefRegTerm(BaselFaceModelManager *pModel) : m_pModel(pModel) {}
    template<typename _Tp>
	bool operator () (_Tp const* const* aParams, _Tp* aResiduals) const {
		const _Tp* aExprCoefs = aParams[0];
		const unsigned int nExprPcs = m_pModel->getNExprPcs();
        for(unsigned int iCoef = 0; iCoef < nExprPcs; iCoef++)
	        aResiduals[iCoef] = _Tp(m_dWeight) * aExprCoefs[iCoef];
		return true;
	}

	static ceres::DynamicAutoDiffCostFunction<ExprCoefRegTerm> *create(BaselFaceModelManager *pModel) {
		return (new ceres::DynamicAutoDiffCostFunction<ExprCoefRegTerm>(new ExprCoefRegTerm(pModel)));
	}

private:
	const double m_dWeight = 0.01;
	BaselFaceModelManager *m_pModel;
};


class ext_parm_reg_term {
public:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	ext_parm_reg_term() {}
    template<typename T>
	bool operator () (const T* const ext_parm, T* residuals) const {
	    residuals[0] = T(u) * ext_parm[0];
		residuals[1] = T(u) * ext_parm[1];
		residuals[2] = T(u) * (ext_parm[2] + 180.0);
	    residuals[3] = T(v) * ext_parm[3];
		residuals[4] = T(v) * ext_parm[4];
		residuals[5] = T(v) * ext_parm[5];
		return true;
	}

	static ceres::CostFunction *create() {
		return (new ceres::AutoDiffCostFunction<ext_parm_reg_term, 6, 6>(
			new ext_parm_reg_term()));
	}
private:
	const double u = 3.0;
	const double v = 0.001;
};

#endif // HPE_REG_TERM_H