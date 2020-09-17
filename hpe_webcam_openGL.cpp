#include "hpe_problem.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <shader.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <getopt.h>

void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

const std::string INPUT_FILE_PATH = "/home/bemfoo/Project/head-pose-estimation/example_inputs/example_inputs_68.txt";
const std::string DEFAULT_OUTPUT_VIDEO_PATH = "output.avi";
const std::string DEFAULT_VIDEO_PATH = "/home/bemfoo/Project/head-pose-estimation/data/person0.avi";
const std::string DLIB_LANDMARK_DETECTOR_DATA_PATH = "/home/bemfoo/Data/shape_predictor_68_face_landmarks.dat";
const char *MODEL_VERTEX_SHADER_PATH = "/home/bemfoo/Project/head-pose-estimation/shader/model.vs";
const char *MODEL_FRAGMENT_SHADER_PATH = "/home/bemfoo/Project/head-pose-estimation/shader/model.fs";
const char *PLANE_VERTEX_SHADER_PATH = "/home/bemfoo/Project/head-pose-estimation/shader/plane.vs";
const char *PLANE_FRAGMENT_SHADER_PATH = "/home/bemfoo/Project/head-pose-estimation/shader/plane.fs";

const unsigned int SCR_WIDTH = 640;
const unsigned int SCR_HEIGHT = 480;


int main(int argc, char** argv)
{  
	google::InitGoogleLogging(argv[0]);
	HeadPoseEstimationProblem *pHpeProblem = new HeadPoseEstimationProblem(INPUT_FILE_PATH);
	BaselFaceModelManager *pBfmManager = pHpeProblem->getModel();

    Eigen::VectorXf tex = pBfmManager->getStdTex().cast<float>();
    Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> tr = pBfmManager->getTriangleList();
    // Triangle list's index begin with 1, need begin with 0
    for(unsigned int iRow = 0; iRow < tr.rows(); iRow++)
        tr[iRow]--; 

    BFM_DEBUG(PRINT_GREEN "#################### OpenGL Init ####################\n" COLOR_END);
    // Initialize and configure GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Head Pose Estimation - Oneshot", NULL, NULL);
    if (window == NULL)
    {
        BFM_ERROR("Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    // Load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        BFM_ERROR("Failed to initialize GLAD\n");
        return -1;
    }

    // Build and compile shader program
    Shader planeShader(PLANE_VERTEX_SHADER_PATH, PLANE_FRAGMENT_SHADER_PATH); // Shader of plane to show photo
    Shader modelShader(MODEL_VERTEX_SHADER_PATH, MODEL_FRAGMENT_SHADER_PATH); // Shader of face model 

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions        // texture coords
         320.0f,  240.0f, 0.0f, 0.0f, 0.0f, // top right
         320.0f, -240.0f, 0.0f, 0.0f, 1.0f, // bottom right
        -320.0f, -240.0f, 0.0f, 1.0f, 1.0f, // bottom left
        -320.0f,  240.0f, 0.0f, 1.0f, 0.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    unsigned int faceVBO, faceVAO, faceEBO;

    glGenVertexArrays(1, &faceVAO);
    glGenBuffers(1, &faceVBO);
    glGenBuffers(1, &faceEBO); 
    glBindVertexArray(faceVAO);  

    int n_vec = pBfmManager->getNVertices() * 3;
    glBindBuffer(GL_ARRAY_BUFFER, faceVBO);
    glBufferData(GL_ARRAY_BUFFER, 2 * n_vec * 4, nullptr, GL_DYNAMIC_DRAW);
    // glBufferSubData(GL_ARRAY_BUFFER, 0, n_vec * 4, face.begin());
    glBufferSubData(GL_ARRAY_BUFFER, n_vec * 4, n_vec * 4, tex.data());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(pBfmManager->getNVertices() * 3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceEBO);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER, 
        pBfmManager->getNFaces() * 3 * sizeof(unsigned int), 
        tr.data(), 
        GL_STATIC_DRAW
    );

    // Load and create a texture
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture); 
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Parse command parameters
	int opt;
    static struct option cmdOptions[] = {
        {"video", optional_argument, nullptr, 'v'},
		{"help", no_argument, nullptr, 'h'},
        {0, 0, 0, 0} 
    };
    cv::VideoCapture cap;
    while ((opt = getopt_long(argc, argv, "v:", cmdOptions, nullptr)) != -1) {
       switch(opt)
	   {
			case 'v':
				if(optarg != nullptr)
				{
					BFM_DEBUG("Input local video: %s\n", optarg);
					cap.open(optarg);
				}
				else
				{
					BFM_DEBUG("Input built-in front camera\n");
					cap.open(0);
				}
				break;
			case 'h':
				BFM_DEBUG("Usage: hpe_webcam [OPTION] ... \n");
				BFM_DEBUG("Estimate head pose for given video stream.\n");
				BFM_DEBUG("\t-v\t--video[=FILE]\tLoad video stream. FILE can be video path (use built-in camera if omitted)\n");
				BFM_DEBUG("\t-h\t--help\tDisplay this help and exit\n");
				return 0;
			default:
				BFM_ERROR("Unrecognized command parameters\n");
				return -1;
	   }
    }

	if (!cap.isOpened()) {
		BFM_ERROR("Unable to connect to camera\n");
		BFM_DEBUG("Try to load default video path: %s\n", DEFAULT_VIDEO_PATH.c_str());
		cap.open(DEFAULT_VIDEO_PATH);
		if(!cap.isOpened())
		{
			BFM_ERROR("Still cannot open. Exit.\n");
			return -1;
		}
		else
		{
			BFM_DEBUG("Open successfully. Continue.\n");
		}
	}	

    cv::VideoWriter vid(
        DEFAULT_OUTPUT_VIDEO_PATH, 
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
        30, 
        cv::Size(SCR_WIDTH, SCR_HEIGHT)
    );

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor sp;
    dlib::deserialize(DLIB_LANDMARK_DETECTOR_DATA_PATH) >> sp;
	BFM_DEBUG("Dlib landmark detector load path: %s\n", DLIB_LANDMARK_DETECTOR_DATA_PATH.c_str());

    cv::namedWindow("Model");

    bool bIsFirstFrame = true;
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        cv::Mat matCurrentCapture;
        if(!cap.read(matCurrentCapture))
            break;
        int width = matCurrentCapture.cols;
        int height = matCurrentCapture.rows;
        GLubyte* pixels;
        int pixel_len = width * height * 3;
        pixels = new GLubyte[pixel_len];
        memcpy(pixels, matCurrentCapture.data, pixel_len * sizeof(char));
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, pixels);
        // glGenerateMipmap(GL_TEXTURE_2D);
        delete[] pixels;

        dlib::cv_image<dlib::bgr_pixel> imgCurrentCapture(matCurrentCapture);
        std::vector<dlib::rectangle> aDets = detector(imgCurrentCapture);
	
        dlib::full_object_detection objDetection = sp(imgCurrentCapture, aDets[0]);
        if (aDets.size() != 0) {
            pHpeProblem->setObservedPoints(&objDetection);

            bool bHasConverged;
            if(bIsFirstFrame)
            {
                auto timeStart = std::chrono::system_clock::now();
                bHasConverged = false;
                while(!bHasConverged)
                {
                    bHasConverged = pHpeProblem->solveExtParams(SolveExtParamsMode_UseCeres | SolveExtParamsMode_UseDlt | SolveExtParamsMode_UseLinearizedRadians);
                    bHasConverged &= pHpeProblem->solveShapeCoef();
                    bHasConverged &= pHpeProblem->solveExprCoef();
                }
                auto timeEnd = std::chrono::system_clock::now();
                auto timeDuration = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart);
                BFM_DEBUG("Solution cost of first frame: %lf Second\n", 
                    double(timeDuration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);             
                bIsFirstFrame = false;
            }
            else
            {
                auto timeStart = std::chrono::system_clock::now();
                bHasConverged = false;
                while(!bHasConverged)
                {
                    bHasConverged = pHpeProblem->solveExtParams(SolveExtParamsMode_UseCeres | SolveExtParamsMode_UseLinearizedRadians);
                    bHasConverged &= pHpeProblem->solveExprCoef();
                }
                auto timeEnd = std::chrono::system_clock::now();
                auto timeDuration = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart);
                BFM_DEBUG("Solution cost of frame: %lf Second\n", 
                    double(timeDuration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);             
                
            }

        }

        pBfmManager->genFace();
        Eigen::VectorXf face = pBfmManager->getCurrentBlendshapeTransformed().cast<float>();

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBufferSubData(GL_ARRAY_BUFFER, 0, n_vec * 4, face.data());
        // bind textures on corresponding texture units
        glBindTexture(GL_TEXTURE_2D, texture);

        // render container
        planeShader.use();
        // create transformations
        glm::mat4 view       = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        glm::mat4 projection = glm::mat4(1.0f);
        glm::mat4 model      = glm::mat4(1.0f);
        // projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        projection = glm::ortho(-320.0f, 320.0f, -240.0f, 240.0f, 0.1f, 100.0f);
        view = glm::lookAt(glm::vec3(0.0, 0.0, 5.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0)); 
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
        // pass transformation matrices to the shader
        planeShader.setMat4("projection", projection); // note: currently we set the projection matrix each frame, but since the projection matrix rarely changes it's often best practice to set it outside the main loop only once.
        planeShader.setMat4("view", view);    
        planeShader.setMat4("model", model);    

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    
        modelShader.use();
        modelShader.setBool("isMirror", true);
        modelShader.setMat4("model", model);           

        glEnable(GL_DEPTH_TEST);
        glClear(GL_DEPTH_BUFFER_BIT);
        glBindVertexArray(faceVAO);
        glDrawElements(GL_TRIANGLES, 93322 * 3, GL_UNSIGNED_INT, 0);
        glDisable(GL_DEPTH_TEST);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);

        if(argc > 1) 
        {
            GLubyte *frame =
                (GLubyte *)malloc(3 * SCR_WIDTH * SCR_HEIGHT * sizeof(GLubyte));
            glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, frame);
            cv::Mat src_frame, dst_frame;
            src_frame = cv::Mat(SCR_HEIGHT, SCR_WIDTH, CV_8UC3, (unsigned char *)frame);
            dst_frame.create(src_frame.rows, src_frame.cols, src_frame.type());
            int rows = src_frame.rows;
            for (int i = 0; i < rows; i++)
                src_frame.row(rows - i - 1).copyTo(dst_frame.row(i));
            vid << dst_frame;
            cv::imshow("Model", dst_frame);
            free(frame);
            cv::waitKey(1000/30);
        }
        
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &faceVAO);
    glDeleteBuffers(1, &faceVBO);
    glDeleteBuffers(1, &faceEBO);
    cap.release();

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}


// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}