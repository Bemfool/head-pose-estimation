#include "bfm.h"

void gen_data(int size, double var);
bfm model("/home/keith/head-pose-estimation/inputs.txt");

int main()
{
	gen_data(10, 0.015); 
    return 0;
}

void gen_data(int size, double var) 
{
    for(int i = 0; i < size; i++) 
    {
        model.generate_random_face();
        model.generate_fp_face();
        
        model.set_rotation(0.0, 0.0, -M_PI);
        std::random_device rd;
    	std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, var);

        std::uniform_real_distribution<double> dis2(-0.02, 0.02);

        std::uniform_real_distribution<double> dis3(-0.001, 0.001);

        ofstream out;
        out.open(std::to_string(var) + "_random_face_" + std::to_string(i) + "-y.txt");
        model.set_rotation(dis2(gen), dis2(gen), -M_PI + dis2(gen));
        while(true)
        {
            double new_yaw = model.get_pitch() + dis(gen);
            std::cout << "new: " << new_yaw << " " << model.get_pitch() << std::endl;
            if(new_yaw > 0.4) break;
            model.set_pitch(new_yaw);
            model.set_roll(model.get_roll() + dis3(gen));
            model.set_yaw(model.get_yaw() + dis3(gen));
            dlib::matrix<double> fp_shape = model.get_fp_current_blendshape_transformed();
            for(int i=0; i<68; i++)
                out << (int)fp_shape(i*3) << " " << (int)fp_shape(i*3+1) << " ";
            out << "\n";
        }
        out.close();

        out.open(std::to_string(var) + "_random_face_" + std::to_string(i) + "+y.txt");
        model.set_rotation(dis2(gen), dis2(gen), -M_PI + dis2(gen));
        while(true)
        {
            double new_yaw = model.get_pitch() - dis(gen);
            if(new_yaw < -0.4) break;
            model.set_pitch(new_yaw);
            model.set_roll(model.get_roll() + dis3(gen));
            model.set_yaw(model.get_yaw() + dis3(gen));
            dlib::matrix<double> fp_shape = model.get_fp_current_blendshape_transformed();
            for(int i=0; i<68; i++)
                out << (int)fp_shape(i*3) << " " << (int)fp_shape(i*3+1) << " ";
            out << "\n";
        }
        out.close();


        out.open(std::to_string(var) + "_random_face_" + std::to_string(i) + "-p.txt");
        model.set_rotation(dis2(gen), dis2(gen), -M_PI + dis2(gen));
        while(true)
        {
            double new_yaw = model.get_roll() + dis(gen);
            if(new_yaw > (-M_PI + 0.4)) break;
            model.set_roll(new_yaw);
            model.set_pitch(model.get_pitch() + dis3(gen));
            model.set_yaw(model.get_yaw() + dis3(gen));
            dlib::matrix<double> fp_shape = model.get_fp_current_blendshape_transformed();
            for(int i=0; i<68; i++)
                out << (int)fp_shape(i*3) << " " << (int)fp_shape(i*3+1) << " ";
            out << "\n";
        }
        out.close();

        out.open(std::to_string(var) + "_random_face_" + std::to_string(i) + "+p.txt");
        model.set_rotation(dis2(gen), dis2(gen), -M_PI + dis2(gen));
        while(true)
        {
            double new_yaw = model.get_roll() - dis(gen);
            if(new_yaw < (-M_PI - 0.4)) break;
            model.set_roll(new_yaw);
            model.set_pitch(model.get_pitch() + dis3(gen));
            model.set_yaw(model.get_yaw() + dis3(gen));            
            dlib::matrix<double> fp_shape = model.get_fp_current_blendshape_transformed();
            for(int i=0; i<68; i++)
                out << (int)fp_shape(i*3) << " " << (int)fp_shape(i*3+1) << " ";
            out << "\n";
        }
        out.close();


        out.open(std::to_string(var) + "_random_face_" + std::to_string(i) + "+r.txt");
        model.set_rotation(dis2(gen), dis2(gen), -M_PI + dis2(gen));
        while(true)
        {
            double new_yaw = model.get_yaw() + dis(gen);
            if(new_yaw > 0.4) break;
            model.set_yaw(new_yaw);
            model.set_pitch(model.get_pitch() + dis3(gen));
            model.set_roll(model.get_roll() + dis3(gen));
            dlib::matrix<double> fp_shape = model.get_fp_current_blendshape_transformed();
            for(int i=0; i<68; i++)
                out << (int)fp_shape(i*3) << " " << (int)fp_shape(i*3+1) << " ";
            out << "\n";
        }
        out.close();

        out.open(std::to_string(var) + "_random_face_" + std::to_string(i) + "-r.txt");
        model.set_rotation(dis2(gen), dis2(gen), -M_PI + dis2(gen));
        while(true)
        {
            double new_yaw = model.get_yaw() - dis(gen);
            if(new_yaw < -0.4) break;
            model.set_yaw(new_yaw);
            model.set_pitch(model.get_pitch() + dis3(gen));
            model.set_roll(model.get_roll() + dis3(gen));
            dlib::matrix<double> fp_shape = model.get_fp_current_blendshape_transformed();
            for(int i=0; i<68; i++)
                out << (int)fp_shape(i*3) << " " << (int)fp_shape(i*3+1) << " ";
            out << "\n";
        }
        out.close();

    }
}