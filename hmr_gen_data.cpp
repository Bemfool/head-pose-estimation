#include "bfm_manager.h"

void GenDataSet(int size, double var);
BaselFaceModelManager *g_pModel;

int main()
{
    std::ifstream in;
	in.open("/home/keith/head-pose-estimation/inputs.txt", std::ios::in);
	if(!in.is_open())
	{
		std::cout << "can't open inputs.txt" << std::endl;
		return;
	}

	std::string strBfmH5Path;
	unsigned int nVertice, nFace, nIdPc, nExprPc;
	std::string strIntParam;
	double aIntParams[4] = { 0.0 };
	std::string strShapeMuH5Path, strShapeEvH5Path, strShapePcH5Path;
	std::string strTexMuH5Path, strTexEvH5Path, strTexPcH5Path;
	std::string strExprMuH5Path, strExprEvH5Path, strExprPcH5Path;
	std::string strTlH5Path;
	unsigned int nFp;
	std::string strFpIdxPath = "";
	in >> strBfmH5Path;
	in >> nVertice >> nFace >> nIdPc >> nExprPc;
	for(auto i = 0; i < 4; i++)
	{
		in >> strIntParam;
		aIntParams[i] = atof(strIntParam.c_str());
	}
	in >> strShapeMuH5Path >> strShapeEvH5Path >>strShapePcH5Path;
	in >> strTexMuH5Path >> strTexEvH5Path >> strTexPcH5Path;
	in >> strExprMuH5Path >> strExprEvH5Path >> strExprPcH5Path;
	in >> strTlH5Path;
	in >> nFp;
	if(nFp != 0) in >> strFpIdxPath;
	in.close();

	g_pModel = new BaselFaceModelManager(
		strBfmH5Path,
		nVertice, nFace, nIdPc, nExprPc,
		aIntParams,
		strShapeMuH5Path, strShapeEvH5Path, strShapePcH5Path,
		strTexMuH5Path, strTexEvH5Path, strTexPcH5Path,
		strExprMuH5Path, strExprEvH5Path, strExprPcH5Path,
		strTlH5Path,
		nFp,
		strFpIdxPath
	);

	GenDataSet(10, 0.015); 
    return 0;
}

void GenDataSet(int size, double var) 
{
    for(int i = 0; i < size; i++) 
    {
        model.generate_random_face();
        model.generate_fp_face();
        
        model.setRotation(0.0, 0.0, -M_PI);
        std::random_device rd;
    	std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, var);

        std::uniform_real_distribution<double> dis2(-0.02, 0.02);

        std::uniform_real_distribution<double> dis3(-0.001, 0.001);

        ofstream out;
        out.open(std::to_string(var) + "_random_face_" + std::to_string(i) + "-y.txt");
        model.setRotation(dis2(gen), dis2(gen), -M_PI + dis2(gen));
        while(true)
        {
            double new_yaw = model.getPitch() + dis(gen);
            std::cout << "new: " << new_yaw << " " << model.getPitch() << std::endl;
            if(new_yaw > 0.4) break;
            model.setPitch(new_yaw);
            model.setRoll(model.getRoll() + dis3(gen));
            model.setYaw(model.getYaw() + dis3(gen));
            dlib::matrix<double> fp_shape = model.getLandmarkCurrentBlendshapeTransformed();
            for(int i=0; i<68; i++)
                out << (int)fp_shape(i*3) << " " << (int)fp_shape(i*3+1) << " ";
            out << "\n";
        }
        out.close();

        out.open(std::to_string(var) + "_random_face_" + std::to_string(i) + "+y.txt");
        model.setRotation(dis2(gen), dis2(gen), -M_PI + dis2(gen));
        while(true)
        {
            double new_yaw = model.getPitch() - dis(gen);
            if(new_yaw < -0.4) break;
            model.setPitch(new_yaw);
            model.setRoll(model.getRoll() + dis3(gen));
            model.setYaw(model.getYaw() + dis3(gen));
            dlib::matrix<double> fp_shape = model.getLandmarkCurrentBlendshapeTransformed();
            for(int i=0; i<68; i++)
                out << (int)fp_shape(i*3) << " " << (int)fp_shape(i*3+1) << " ";
            out << "\n";
        }
        out.close();


        out.open(std::to_string(var) + "_random_face_" + std::to_string(i) + "-p.txt");
        model.setRotation(dis2(gen), dis2(gen), -M_PI + dis2(gen));
        while(true)
        {
            double new_yaw = model.getRoll() + dis(gen);
            if(new_yaw > (-M_PI + 0.4)) break;
            model.setRoll(new_yaw);
            model.setPitch(model.getPitch() + dis3(gen));
            model.setYaw(model.getYaw() + dis3(gen));
            dlib::matrix<double> fp_shape = model.getLandmarkCurrentBlendshapeTransformed();
            for(int i=0; i<68; i++)
                out << (int)fp_shape(i*3) << " " << (int)fp_shape(i*3+1) << " ";
            out << "\n";
        }
        out.close();

        out.open(std::to_string(var) + "_random_face_" + std::to_string(i) + "+p.txt");
        model.setRotation(dis2(gen), dis2(gen), -M_PI + dis2(gen));
        while(true)
        {
            double new_yaw = model.getRoll() - dis(gen);
            if(new_yaw < (-M_PI - 0.4)) break;
            model.setRoll(new_yaw);
            model.setPitch(model.getPitch() + dis3(gen));
            model.setYaw(model.getYaw() + dis3(gen));            
            dlib::matrix<double> fp_shape = model.getLandmarkCurrentBlendshapeTransformed();
            for(int i=0; i<68; i++)
                out << (int)fp_shape(i*3) << " " << (int)fp_shape(i*3+1) << " ";
            out << "\n";
        }
        out.close();


        out.open(std::to_string(var) + "_random_face_" + std::to_string(i) + "+r.txt");
        model.setRotation(dis2(gen), dis2(gen), -M_PI + dis2(gen));
        while(true)
        {
            double new_yaw = model.getYaw() + dis(gen);
            if(new_yaw > 0.4) break;
            model.setYaw(new_yaw);
            model.setPitch(model.getPitch() + dis3(gen));
            model.setRoll(model.getRoll() + dis3(gen));
            dlib::matrix<double> fp_shape = model.getLandmarkCurrentBlendshapeTransformed();
            for(int i=0; i<68; i++)
                out << (int)fp_shape(i*3) << " " << (int)fp_shape(i*3+1) << " ";
            out << "\n";
        }
        out.close();

        out.open(std::to_string(var) + "_random_face_" + std::to_string(i) + "-r.txt");
        model.setRotation(dis2(gen), dis2(gen), -M_PI + dis2(gen));
        while(true)
        {
            double new_yaw = model.getYaw() - dis(gen);
            if(new_yaw < -0.4) break;
            model.setYaw(new_yaw);
            model.setPitch(model.getPitch() + dis3(gen));
            model.setRoll(model.getRoll() + dis3(gen));
            dlib::matrix<double> fp_shape = model.getLandmarkCurrentBlendshapeTransformed();
            for(int i=0; i<68; i++)
                out << (int)fp_shape(i*3) << " " << (int)fp_shape(i*3+1) << " ";
            out << "\n";
        }
        out.close();

    }
}