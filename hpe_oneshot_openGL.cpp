#include "hpe_problem.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <shader.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 640;
const unsigned int SCR_HEIGHT = 480;

/* If use different input file, 
 * please update corresponding parameters in include/db_params.h 
 */
const std::string INPUT_FILE_PATH = "/home/bemfoo/Project/head-pose-estimation/example_inputs/example_inputs_68.txt";
// const std::string INPUT_FILE_PATH = "/home/bemfoo/Project/head-pose-estimation/example_inputs/example_inputs_6.txt";
// const std::string INPUT_FILE_PATH = "/home/bemfoo/Project/head-pose-estimation/example_inputs/example_inputs_12.txt";
const std::string DEFAULT_IMG_PATH = "/home/bemfoo/Project/head-pose-estimation/data/profile0.jpg";
const std::string DLIB_LANDMARK_DETECTOR_DATA_PATH = "/home/bemfoo/Data/shape_predictor_68_face_landmarks.dat";
const char *MODEL_VERTEX_SHADER_PATH = "/home/bemfoo/Project/head-pose-estimation/shader/model.vs";
const char *MODEL_FRAGMENT_SHADER_PATH = "/home/bemfoo/Project/head-pose-estimation/shader/model.fs";
const char *PLANE_VERTEX_SHADER_PATH = "/home/bemfoo/Project/head-pose-estimation/shader/plane.vs";
const char *PLANE_FRAGMENT_SHADER_PATH = "/home/bemfoo/Project/head-pose-estimation/shader/plane.fs";


int main(int argc, char** argv)
{  
	google::InitGoogleLogging(argv[0]);

	/* Init head pose estimation problem */
	HeadPoseEstimationProblem *pHpeProblem = new HeadPoseEstimationProblem(INPUT_FILE_PATH);
	BaselFaceModelManager *pBfmManager = pHpeProblem->getModel();

	dlib::array2d<dlib::rgb_pixel> arr2dImg;
	std::string strImgName = DEFAULT_IMG_PATH;
	if(argc > 1) strImgName = argv[1];	
	BFM_DEBUG("Image to be processed: %s\n", strImgName.c_str());
	load_image(arr2dImg, strImgName);	

	try 
	{
		// Init detector
		dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
		dlib::shape_predictor sp;
		dlib::deserialize(DLIB_LANDMARK_DETECTOR_DATA_PATH) >> sp;
        BFM_DEBUG("Detector init successfully\n");

		// pyramid_up(img);
		std::vector<dlib::rectangle> dets = detector(arr2dImg);

        /* Only detect the first face */
		if(dets.size() != 0) 
		{
            /* Load landmarks detected by Dlib */
			dlib::full_object_detection objDetection = sp(arr2dImg, dets[0]);
			pHpeProblem->setObservedPoints(&objDetection);

            // Start of solving 
            auto start = std::chrono::system_clock::now();
            
            // There are some different ways to choose to solve external parameters 
			if(argc > 2)
				pHpeProblem->solveExtParams(
					SolveExtParamsMode_UseCeres | SolveExtParamsMode_UseDlt | SolveExtParamsMode_UseLinearizedRadians, 
					atof(argv[2]), atof(argv[3]));
			else
				pHpeProblem->solveExtParams(SolveExtParamsMode_UseCeres | SolveExtParamsMode_UseDlt | SolveExtParamsMode_UseLinearizedRadians);
			// pHpeProblem->solveExtParams(SolveExtParamsMode_UseOpenCV);

			pHpeProblem->solveShapeCoef();
			pHpeProblem->solveExprCoef();
			
			// End of solving
			auto end = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			BFM_DEBUG("Cost of solution: %lf Second\n", 
				double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);             
            pBfmManager->printExtParams();            

            // Show results 
            pBfmManager->printExtParams();
			// pBfmManager->printShapeCoef();
			// pBfmManager->printExprCoef();

            // Generate whole face, because before functions only process landmarks
			pBfmManager->genFace();

            // Write face into .ply model file 
			// pBfmManager->writePly("rnd_face.ply", (ModelWriteMode_CameraCoord | ModelWriteMode_PickLandmark));

		}

	} catch (exception& e) {
        BFM_ERROR("Exception thrown: %s\n", e.what());
	}

    BFM_DEBUG(PRINT_GREEN "#################### OpenGL Init ####################\n" COLOR_END);
    // Initialize and configure GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); 
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Head Pose Estimation - Oneshot", nullptr, nullptr);
    if (window == nullptr)
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

    // Set up vertex data and buffers and configure vertex attributes
    float vertices[] = {
        // Positions            Texture coords
         320.0f,  240.0f, 0.0f, 1.0f, 1.0f, // Top right
         320.0f, -240.0f, 0.0f, 1.0f, 0.0f, // Bottom right 
        -320.0f, -240.0f, 0.0f, 0.0f, 0.0f, // Bottom left 
        -320.0f,  240.0f, 0.0f, 0.0f, 1.0f  // Top left
    };

    unsigned int indices[] = {
        0, 1, 3, // First triangle
        1, 2, 3  // Second triangle
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

    // Position attribute 
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    Eigen::VectorXf face = pBfmManager->getCurrentBlendshapeTransformed().cast<float>();
    Eigen::VectorXf tex = pBfmManager->getStdTex().cast<float>();
    Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> tr = pBfmManager->getTriangleList();
    // Triangle list's index begin with 1, need begin with 0
    for(unsigned int iRow = 0; iRow < tr.rows(); iRow++)
        tr[iRow]--; 

    unsigned int faceVBO, faceVAO, faceEBO;

    glGenVertexArrays(1, &faceVAO);
    glGenBuffers(1, &faceVBO);
    glGenBuffers(1, &faceEBO); 
    glBindVertexArray(faceVAO);  

    int n_vec = pBfmManager->getNVertices() * 3;
    glBindBuffer(GL_ARRAY_BUFFER, faceVBO);
    glBufferData(GL_ARRAY_BUFFER, 2 * n_vec * 4, nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, n_vec * 4, face.data());
    glBufferSubData(GL_ARRAY_BUFFER, n_vec * 4, n_vec * 4, tex.data());

    /* Position attribute */
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    /* Texture attribute */
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

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true); /* Flip loaded texture's on the y-axis */
    unsigned char *data = stbi_load(strImgName.c_str(), &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    planeShader.use();
    glm::mat4 view       = glm::mat4(1.0f); 
    glm::mat4 projection = glm::mat4(1.0f);
    glm::mat4 model      = glm::mat4(1.0f);
    projection = glm::ortho(-320.0f, 320.0f, -240.0f, 240.0f, 0.1f, 100.0f);
    view = glm::lookAt(glm::vec3(0.0, 0.0, 5.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0)); 
    model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
    planeShader.setMat4("projection", projection); 
    planeShader.setMat4("view", view);    
    planeShader.setMat4("model", model);

    modelShader.use();
    model = glm::mat4(1.0f); 
    modelShader.setMat4("model", model);           
    modelShader.setBool("isMirror", false);

    /* Render loop */
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, texture);

        /* Draw photo */
        planeShader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        /* Draw face */
        glEnable(GL_DEPTH_TEST);
        glClear(GL_DEPTH_BUFFER_BIT);
        modelShader.use();
        glBindVertexArray(faceVAO);
        glDrawElements(GL_TRIANGLES, 93322 * 3, GL_UNSIGNED_INT, 0);
        glDisable(GL_DEPTH_TEST);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &faceVAO);
    glDeleteBuffers(1, &faceVBO);
    glDeleteBuffers(1, &faceEBO);

    glfwTerminate();
    return 0;
}


void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}


void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}