#include "hpe_problem.h"

HeadPoseEstimationProblem::HeadPoseEstimationProblem(std::string strFilename) {
	init(strFilename);
}

BfmStatus HeadPoseEstimationProblem::init(std::string strFilename) {
	BFM_DEBUG(PRINT_GREEN "#################### Init Head Pose Esitimation Problem ####################\n" COLOR_END);

	std::ifstream in;
	in.open(strFilename, std::ios::in);
	if(!in.is_open())
	{
		BFM_ERROR("Cannot open %s\n", strFilename.c_str());
		return BfmStatus_Error;
	}
	
	std::string strBfmH5Path;
	unsigned int nVertices, nFaces, nIdPcs, nExprPcs;
	std::string strIntParams;
	double aIntParams[4] = { 0.0 };
	std::string strShapeMuH5Path, strShapeEvH5Path, strShapePcH5Path;
	std::string strTexMuH5Path, strTexEvH5Path, strTexPcH5Path;
	std::string strExprMuH5Path, strExprEvH5Path, strExprPcH5Path;
	std::string strTriangleListH5Path;
	unsigned int nLandmarks;
	std::string strLandmarkIdxPath = "", strLandmarkMapPath;
	in >> strBfmH5Path;
	in >> nVertices >> nFaces >> nIdPcs >> nExprPcs;
	for(auto i = 0; i < 4; i++)
	{
		in >> strIntParams;
		aIntParams[i] = atof(strIntParams.c_str());
	}
	in >> strShapeMuH5Path >> strShapeEvH5Path >>strShapePcH5Path;
	in >> strTexMuH5Path >> strTexEvH5Path >> strTexPcH5Path;
	in >> strExprMuH5Path >> strExprEvH5Path >> strExprPcH5Path;
	in >> strTriangleListH5Path;
	in >> nLandmarks;
	if(nLandmarks != 0)
	{
		in >> strLandmarkIdxPath >> strLandmarkMapPath;
		std::ifstream tmpFile;
		tmpFile.open(strLandmarkMapPath, std::ios::in);
		if(!tmpFile.is_open())
		{
			BFM_ERROR("Cannot open %s\n", strLandmarkMapPath.c_str());
			return BfmStatus_Error;
		}

		for(unsigned int iLandmark = 0; iLandmark < nLandmarks; iLandmark++)
		{
			int iLandmarkMapIdx;
			tmpFile >> iLandmarkMapIdx;
			m_aLandmarkMap.push_back(iLandmarkMapIdx);
		}
		tmpFile.close();
	}
	
	in.close();

	m_pModel = new BaselFaceModelManager(
		strBfmH5Path,
		nVertices, nFaces, nIdPcs, nExprPcs,
		aIntParams,
		strShapeMuH5Path, strShapeEvH5Path, strShapePcH5Path,
		strTexMuH5Path, strTexEvH5Path, strTexPcH5Path,
		strExprMuH5Path, strExprEvH5Path, strExprPcH5Path,
		strTriangleListH5Path,
		nLandmarks,
		strLandmarkIdxPath
	);

	BFM_DEBUG("Load model path: \t%s\n", strBfmH5Path.c_str());
	BFM_DEBUG("Number of vertices: \t%u\n", nVertices);
	BFM_DEBUG("Number of faces: \t%u\n", nFaces);
	BFM_DEBUG("Number of identity principle component: \t%u\n", nIdPcs);
	BFM_DEBUG("Number of expression principle component: \t%u\n", nExprPcs);
	BFM_DEBUG("Camera intrinsic parameters: \n");
	BFM_DEBUG("	fx: \t%.2lf", aIntParams[0]);
	BFM_DEBUG("	fy: \t%.2lf", aIntParams[1]);
	BFM_DEBUG("	cx: \t%.2lf", aIntParams[2]);
	BFM_DEBUG("	cy: \t%.2lf\n", aIntParams[3]);
	BFM_DEBUG("Number of landmarks: \t%u\n\n", nLandmarks);

	return BfmStatus_Ok;
}


void HeadPoseEstimationProblem::dlt()
{
	BFM_DEBUG("Calculating extrinsic parameters using Direct Linear Transform.\n");
	unsigned int nLandmarks = m_pModel->getNLandmarks();
	assert(nLandmarks >= 6);

	cv::Ptr<CvMat> matL;
	double* L;
	double LL[12 * 12], LW[12], LV[12 * 12];
	CvMat _LL = cvMat(12, 12, CV_64F, LL);
	CvMat _LW = cvMat(12, 1, CV_64F, LW);
	CvMat _LV = cvMat(12, 12, CV_64F, LV);
	CvMat _RRt, _RR, _tt;

	matL.reset(cvCreateMat(2 * nLandmarks, 12, CV_64F));
	L = matL->data.db;
	const VectorXd &vecLandmarkBlendshape = m_pModel->getLandmarkCurrentBlendshape();

	const double fx = m_pModel->getFx();
	const double fy = m_pModel->getFy();
	const double cx = m_pModel->getCx();
	const double cy = m_pModel->getCy();

	for(int i = 0; i < N_LANDMARK; i++, L += 24)
	{
		double x = m_pObservedPoints->part(i).x(), y = m_pObservedPoints->part(i).y();
		double X = vecLandmarkBlendshape(i * 3);
		double Y = vecLandmarkBlendshape(i * 3 + 1);
		double Z = vecLandmarkBlendshape(i * 3 + 2);
		L[0] = X * fx; L[16] = X * fy;
		L[1] = Y * fx; L[17] = Y * fy;
		L[2] = Z * fx; L[18] = Z * fy;
		L[3] = fx; L[19] = fy;
		L[4] = L[5] = L[6] = L[7] = 0.;
		L[12] = L[13] = L[14] = L[15] = 0.;
		L[8] = X * cx - x * X;
		L[9] = Y * cx - x * Y;
		L[10] = Z * cx - x * Z;
		L[11] = cx - x;
		L[20] = X * cy - y * X;
		L[21] = Y * cy - y * Y;
		L[22] = Z * cy - y * Z;
		L[23] = cy - y;
	}

	cvMulTransposed(matL, &_LL, 1);
	cvSVD(&_LL, &_LW, 0, &_LV, CV_SVD_MODIFY_A + CV_SVD_V_T);
	_RRt = cvMat(3, 4, CV_64F, LV + 11 * 12);
	cvGetCols(&_RRt, &_RR, 0, 3);
	cvGetCol(&_RRt, &_tt, 3);

	m_pModel->setMatR(&_RR);
	m_pModel->setVecT(&_tt);
	m_pModel->genExtParams();

}


bool HeadPoseEstimationProblem::solveExtParams(long mode, double ca, double cb) 
{
	BFM_DEBUG(PRINT_GREEN "#################### Estimate Extrinsic Parameters ####################\n" COLOR_END);
	if(mode & SolveExtParamsMode_UseOpenCV)
	{
		const double *aIntParams = m_pModel->getIntParams();
		double aCameraMatrix[3][3] = {
			{aIntParams[0], 0.0, aIntParams[2]},
			{0.0, aIntParams[1], aIntParams[3]},
			{0.0, 0.0, 1.0}
		};
		cv::Mat matCameraMatrix = cv::Mat(3, 3, CV_64FC1, aCameraMatrix);		
		#ifdef _DEBUG
		std::cout << "camera matrix: " << matCameraMatrix << std::endl;
		#endif
		std::vector<float> aDistCoef(0);
		std::vector<cv::Point3f> out;
		std::vector<cv::Point2f> in;
		cv::Mat rvec, tvec;
		VectorXd vecLandmarkBlendshape = m_pModel->getLandmarkCurrentBlendshape();
		for(unsigned int iLandmark = 0; iLandmark < 68; iLandmark++) {
			out.push_back(cv::Point3f(
				vecLandmarkBlendshape(iLandmark * 3), 
				vecLandmarkBlendshape(iLandmark * 3 + 1), 
				vecLandmarkBlendshape(iLandmark * 3 + 2)));
			in.push_back(cv::Point2f(m_pObservedPoints->part(iLandmark).x(), m_pObservedPoints->part(iLandmark).y()));
		}
		cv::solvePnP(out, in, matCameraMatrix, aDistCoef, rvec, tvec);
		cv::Rodrigues(rvec, rvec);
		#ifdef _DEBUG
		std::cout << rvec << std::endl;
		std::cout << tvec << std::endl;
		#endif
		m_pModel->setMatR(rvec);
		m_pModel->setVecT(tvec);
		m_pModel->genExtParams();
		return true;
	}
	else if(mode & SolveExtParamsMode_UseLinearizedRadians)
	{
		BFM_DEBUG("solve -> external parameters (linealized)\n");
		if(mode & SolveExtParamsMode_UseDlt)
		{
			BFM_DEBUG("	1) estimate initial values by using DLT algorithm.\n");
			dlt();
		}
		else
		{
			BFM_DEBUG("	1) initial values have been set in advance or are 0s.\n");
		}
		
		m_pModel->genTransMat();	

		CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
		while(true) 
		{
			ceres::Problem problem;
			double aSmallExtParams[6] = { 0.f };
			ceres::CostFunction *costFunction = LinearizedExtParamsReprojErr::create(m_pObservedPoints, m_pModel, m_aLandmarkMap, ca, cb);
			problem.AddResidualBlock(costFunction, nullptr, aSmallExtParams);
			ceres::Solve(options, &problem, &summary);
			BFM_DEBUG("%s\n", summary.BriefReport().c_str());
			
			if(is_close_enough(aSmallExtParams, 0, 0)) 
			{
				#ifdef _DEBUG
				bfm_utils::PrintArr(aSmallExtParams, 6);
				std::cout << summary.BriefReport() << std::endl;
				#endif
				break; 
			}

			m_pModel->accExtParams(aSmallExtParams);
			
			#ifdef _DEBUG
			bfm_utils::PrintArr(aSmallExtParams, 6);							
			#endif
		}
		m_pModel->genExtParams();
		return (summary.termination_type == ceres::CONVERGENCE);
	}
	else
	{
		#ifndef HPE_SHUT_UP
		std::cout << "solve -> external parameters" << std::endl;	
		std::cout << "init ceres solve - ";
		#endif

		#ifndef HPE_SHUT_UP
		if(mode & SolveExtParamsMode_UseDlt)
		{
			std::cout << "	1) esitimate initial values by using DLT algorithm." << std::endl;
			dlt();
		}
		else
		{
			std::cout << "	1) initial values have been set in advance or are 0s." << std::endl;
		}
		#else
		if(mode & SolveExtParamsMode_UseDlt) dlt();
		#endif

		ceres::Problem problem;
		double *ext_params = m_pModel->getMutableExtParams();
		ceres::CostFunction *costFunction = ExtParamsReprojErr::create(m_pObservedPoints, m_pModel, m_aLandmarkMap);
		problem.AddResidualBlock(costFunction, nullptr, ext_params);
		ceres::Solver::Options options;
		options.max_num_iterations = 100;
		options.num_threads = 8;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		#ifndef HPE_SHUT_UP
		std::cout << "success" << std::endl;
		#endif
		ceres::Solve(options, &problem, &summary);
		#ifndef HPE_SHUT_UP
		std::cout << summary.BriefReport() << std::endl;
		#endif
		m_pModel->genTransMat();
		return (summary.termination_type == ceres::CONVERGENCE);
	}
}


bool HeadPoseEstimationProblem::solveShapeCoef() {
	BFM_DEBUG(PRINT_GREEN "#################### Estimate Shape Coefficients ####################\n" COLOR_END);
	ceres::Problem problem;
	double *aShapeCoef = m_pModel->getMutableShapeCoef();
	ceres::CostFunction *costFunction = ShapeCoefReprojErr::create(m_pObservedPoints, m_pModel, m_aLandmarkMap);
	ceres::CostFunction *regTerm = shape_coef_reg_term::create();
	problem.AddResidualBlock(costFunction, nullptr, aShapeCoef);
	problem.AddResidualBlock(regTerm, nullptr, aShapeCoef);
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	ceres::Solve(options, &problem, &summary);
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	m_pModel->genLandmarkFace();
	return (summary.termination_type == ceres::CONVERGENCE);
}


bool HeadPoseEstimationProblem::solveExprCoef() {
	BFM_DEBUG(PRINT_GREEN "#################### Estimate Expression Coefficients ####################\n" COLOR_END);
	ceres::Problem problem;
	double *aExprCoef = m_pModel->getMutableExprCoef();
	ceres::CostFunction *costFunction = ExprCoefReprojErr::create(m_pObservedPoints, m_pModel, m_aLandmarkMap);
	ceres::CostFunction *regTerm = expr_coef_reg_term::create();
	problem.AddResidualBlock(costFunction, nullptr, aExprCoef);
	problem.AddResidualBlock(regTerm, nullptr, aExprCoef);
	CERES_INIT(N_CERES_ITERATIONS, N_CERES_THREADS, B_CERES_STDCOUT);
	ceres::Solve(options, &problem, &summary);
	BFM_DEBUG("%s\n", summary.BriefReport().c_str());
	m_pModel->genLandmarkFace();
	return (summary.termination_type == ceres::CONVERGENCE);
}


bool HeadPoseEstimationProblem::is_close_enough(double *ext_params, double rotation_eps, double translation_eps)
{
	for(int i=0; i<3; i++)
		if(abs(ext_params[i]) > rotation_eps)
			return false;
	for(int i=3; i<6; i++)
		if(abs(ext_params[i]) > translation_eps)
			return false;
	return true;
}


