#include "hpe.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <shader.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 640;
const unsigned int SCR_HEIGHT = 480;

using namespace dlib;
using namespace std;

int main(int argc, char** argv)
{  
	google::InitGoogleLogging(argv[0]);
	/* Init head pose estimation problem */
	hpe hpe_problem("../example_inputs/example_inputs_68.txt");

	/* If use different input file, 
	 * please update corresponding parameters in include/db_params.h 
	 */
	// hpe hpe_problem("../example_inputs/example_inputs_6.txt");
	// hpe hpe_problem("../example_inputs/example_inputs_12.txt");
	// hpe hpe_problem("../example_inputs/example_inputs_21.txt");
	// hpe hpe_problem("../example_inputs/example_inputs_33.txt");

	/* Open image with human face (default: test.jpg) */
	array2d<rgb_pixel> img;
	std::string img_name = "test.jpg";
	if(argc > 1) img_name = argv[1];	
	std::cout << "Processing image: " << img_name << std::endl;
	load_image(img, img_name);	

	try 
	{
		/* Init detector */
		dlib::frontal_face_detector detector = get_frontal_face_detector();
		dlib::shape_predictor sp;
		deserialize("../data/shape_predictor_68_face_landmarks.dat") >> sp;
		std::cout << "Detector init successfully\n" << std::endl;

		// pyramid_up(img);
		std::vector<dlib::rectangle> dets = detector(img);
		std::cout << "Number of faces detected: " << dets.size() << std::endl;

        /* Only detect the first face */
		if(dets.size() != 0) 
		{
            /* Load landmarks detected by Dlib */
			full_object_detection obj_detection = sp(img, dets[0]);
			hpe_problem.set_observed_points(obj_detection);

            /* Start of solving */
            auto start = std::chrono::system_clock::now();
            
            /* There are some different ways to choose to solve external parameters */
			std::cout << "Solving external parameers..." << std::endl;
			hpe_problem.solve_ext_params(USE_CERES);

			// if(argc > 2)
			// 	hpe_problem.solve_ext_params(USE_CERES | USE_DLT | USE_LINEARIZED_RADIANS, atof(argv[2]), atof(argv[3]));
			// else
			// 	hpe_problem.solve_ext_params(USE_CERES | USE_DLT | USE_LINEARIZED_RADIANS);

			std::cout << "Solving shape coefficients..." << std::endl;
			hpe_problem.solve_shape_coef();

			std::cout << "Solving expression coefficients..." << std::endl;
			hpe_problem.solve_expr_coef();
			
			/* End of solving */
			auto end = std::chrono::system_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			std::cout << "Solve cost: " << double(duration.count())
				* std::chrono::microseconds::period::num
				/ std::chrono::microseconds::period::den << " Seconds\n" << std::endl;                

            /* Show results */
            hpe_problem.get_model().print_extrinsic_params();
			// hpe_problem.get_model().print_intrinsic_params();
			// hpe_problem.get_model().print_shape_coef();
			// hpe_problem.get_model().print_expr_coef();

            /* Generate whole face, because before functions only process landmarks */
			hpe_problem.get_model().generate_face();

            /* Write face into .ply model file */
			// hpe_problem.get_model().ply_write("rnd_face.ply", (CAMERA_COORD | PICK_FP));
		}

	} catch (exception& e) {
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}

    /* GLFW: Initialize and configure */
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); /* Uncomment this statement to fix compilation on OS X */
#endif

    /* GLFW window creation */
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Head Pose Estimation - Oneshot", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    /* GLAD: Load all OpenGL function pointers */
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    /* Build and compile shader program */
    Shader planeShader("../shader/plane.vs", "../shader/plane.fs"); /* Shader of plane to show photo */
    Shader modelShader("../shader/model.vs", "../shader/model.fs"); /* Shader of face model */

    /* Set up vertex data and buffers and configure vertex attributes */
    float vertices[] = {
        /* Positions            Texture coords */
         320.0f,  240.0f, 0.0f, 1.0f, 1.0f, /* Top right */
         320.0f, -240.0f, 0.0f, 1.0f, 0.0f, /* Bottom right */
        -320.0f, -240.0f, 0.0f, 0.0f, 0.0f, /* Bottom left */
        -320.0f,  240.0f, 0.0f, 0.0f, 1.0f  /* Top left */ 
    };

    unsigned int indices[] = {
        0, 1, 3, /* First triangle */
        1, 2, 3  /* Second triangle */
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

    /* Position attribute */
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    /* Texture coord attribute */
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    dlib::matrix<float> face = dlib::matrix_cast<float>(hpe_problem.get_model().get_current_blendshape_transformed());
    dlib::matrix<float> tex = dlib::matrix_cast<float>(hpe_problem.get_model().get_std_tex());
    dlib::matrix<unsigned int> tr = dlib::matrix_cast<unsigned int>(hpe_problem.get_model().get_tl());
    for(auto i = tr.begin(); i != tr.end(); i++) 
        *i = (*i) - 1;  /* Triangle list's index begin with 1, need begin with 0 */

    unsigned int faceVBO, faceVAO, faceEBO;

    glGenVertexArrays(1, &faceVAO);
    glGenBuffers(1, &faceVBO);
    glGenBuffers(1, &faceEBO); 
    glBindVertexArray(faceVAO);  

    int n_vec = hpe_problem.get_model().get_n_vertice() * 3;
    glBindBuffer(GL_ARRAY_BUFFER, faceVBO);
    glBufferData(GL_ARRAY_BUFFER, 2 * n_vec * 4, nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, n_vec * 4, face.begin());
    glBufferSubData(GL_ARRAY_BUFFER, n_vec * 4, n_vec * 4, tex.begin());

    /* Position attribute */
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    /* Texture attribute */
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(46990 * 3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceEBO);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER, 
        hpe_problem.get_model().get_n_face() * 3 * sizeof(unsigned int), 
        tr.begin(), 
        GL_STATIC_DRAW
    );

    /* Load and create a texture */
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture); 

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true); /* Flip loaded texture's on the y-axis */
    unsigned char *data = stbi_load(img_name.c_str(), &width, &height, &nrChannels, 0);
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


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}