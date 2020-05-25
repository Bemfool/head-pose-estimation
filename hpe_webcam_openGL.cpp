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
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

using namespace dlib;
using namespace std;

int main(int argc, char** argv)
{  
	google::InitGoogleLogging(argv[0]);
	hpe hpe_problem("/home/keith/head-pose-estimation/inputs.txt");

    dlib::matrix<float> tex = dlib::matrix_cast<float>(hpe_problem.get_model().get_std_tex());
    dlib::matrix<unsigned int> tr = dlib::matrix_cast<unsigned int>(hpe_problem.get_model().get_tl());
    for(auto i = tr.begin(); i != tr.end(); i++) 
    {
        *i = (*i) - 1;
    }

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Head Pose Estimation - Oneshot", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // build and compile our shader zprogram
    // ------------------------------------
    Shader ourShader("../shader/plane.vs", "../shader/plane.fs");
    Shader modelShader("../shader/model.vs", "../shader/model.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions        // texture coords
         400.0f,  300.0f, 0.0f, 0.0f, 0.0f, // top right
         400.0f, -300.0f, 0.0f, 0.0f, 1.0f, // bottom right
        -400.0f, -300.0f, 0.0f, 1.0f, 1.0f, // bottom left
        -400.0f,  300.0f, 0.0f, 1.0f, 0.0f  // top left 
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

    int n_vec = hpe_problem.get_model().get_n_vertice() * 3;
    glBindBuffer(GL_ARRAY_BUFFER, faceVBO);
    glBufferData(GL_ARRAY_BUFFER, 2 * n_vec * 4, nullptr, GL_DYNAMIC_DRAW);
    // glBufferSubData(GL_ARRAY_BUFFER, 0, n_vec * 4, face.begin());
    glBufferSubData(GL_ARRAY_BUFFER, n_vec * 4, n_vec * 4, tex.begin());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(46990 * 3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    int n_tl = hpe_problem.get_model().get_n_face() * 3;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, n_tl * sizeof(unsigned int), tr.begin(), GL_STATIC_DRAW);


    // load and create a texture 
    // -------------------------
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture); 
     // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    cv::VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened()) {
        std::cerr << "Unable to connect to camera" << std::endl;
        return -1;
    }	

    // Init Detector
    std::cout << "initing detector..." << std::endl;
    dlib::frontal_face_detector detector = get_frontal_face_detector();
    dlib::shape_predictor sp;
    deserialize("../data/shape_predictor_68_face_landmarks.dat") >> sp;
    std::cout << "detector init successfully\n" << std::endl;

    // render loop
    // -----------
    bool is_first_frame = true;
    while (!glfwWindowShouldClose(window))
    {

        // input
        // -----
        processInput(window);

        cv::Mat f;
        if(!cap.read(f))
            break;
        int width = f.cols;
        int height = f.rows;
        GLubyte* pixels;
        int pixel_len = width * height * 3;
        pixels = new GLubyte[pixel_len];
        memcpy(pixels, f.data, pixel_len * sizeof(char));
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, pixels);
        // glGenerateMipmap(GL_TEXTURE_2D);
        delete[] pixels;

        dlib::cv_image<bgr_pixel> img(f);
    
        std::vector<rectangle> dets = detector(img);
			// cout << "Number of faces detected: " << dets.size() << endl;
	
        std::vector<dlib::full_object_detection> obj_detections;
        if (dets.size() > 0) {
            // 将当前获得的特征点数据放置到全局
            full_object_detection obj_detection = sp(img, dets[0]);
            obj_detections.push_back(obj_detection);
            hpe_problem.set_observed_points(obj_detection);

            bool state;
            if(is_first_frame)
            {
                auto start = std::chrono::system_clock::now();
                state = false;
                while(!state)
                {
                    state = hpe_problem.solve_ext_params(USE_LINEARIZED_RADIANS | USE_DLT);
                    state &= hpe_problem.solve_shape_coef();
                    state &= hpe_problem.solve_expr_coef();
                }
                auto end = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "First frame cost: " << double(duration.count())
                    * std::chrono::microseconds::period::num
                    / std::chrono::microseconds::period::den << " Seconds\n" << std::endl;                

                is_first_frame = false;
            }
            else
            {
                auto start = std::chrono::system_clock::now();
                state = false;
                while(!state)
                {
                    state = hpe_problem.solve_ext_params(USE_LINEARIZED_RADIANS);
                    state &= hpe_problem.solve_expr_coef();
                }
                auto end = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "First frame cost: " << double(duration.count())
                    * std::chrono::microseconds::period::num
                    / std::chrono::microseconds::period::den << " Seconds\n" << std::endl;                

            }

        }
        hpe_problem.get_model().generate_face();
        std::cout << hpe_problem.get_model().get_mutable_shape_coef()[0] << std::endl;
        std::cout << hpe_problem.get_model().get_mutable_expr_coef()[0] << std::endl;
        dlib::matrix<float> face = dlib::matrix_cast<float>(hpe_problem.get_model().get_current_blendshape());

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBufferSubData(GL_ARRAY_BUFFER, 0, n_vec * 4, face.begin());
        // bind textures on corresponding texture units
        glBindTexture(GL_TEXTURE_2D, texture);

        // render container
        ourShader.use();
        // create transformations
        glm::mat4 view       = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        glm::mat4 projection = glm::mat4(1.0f);
        glm::mat4 model      = glm::mat4(1.0f);
        // projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        projection = glm::ortho(-400.0f, 400.0f, -300.0f, 300.0f, 0.1f, 100.0f);
        view = glm::lookAt(glm::vec3(0.0, 0.0, 5.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0)); 
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
        // pass transformation matrices to the shader
        ourShader.setMat4("projection", projection); // note: currently we set the projection matrix each frame, but since the projection matrix rarely changes it's often best practice to set it outside the main loop only once.
        ourShader.setMat4("view", view);    
        ourShader.setMat4("model", model);    

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    
        modelShader.use();
        view       = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        projection = glm::mat4(1.0f);
        model      = glm::mat4(1.0f);

        projection = glm::perspective(
            glm::radians(60.0f), 
            (float)SCR_WIDTH / (float)SCR_HEIGHT, 
            0.1f, 
            20000000.0f);
        view = glm::lookAt(
            glm::vec3(0.0, 0.0, -500000), 
            glm::vec3(0.0, 0.0, 0.0), 
            glm::vec3(0.0, -1.0, 0.0)); 

        model = glm::scale(model, glm::vec3(-1.0, 1.0, 1.0));
        model = glm::rotate(model, (float)hpe_problem.get_model().get_yaw(), glm::vec3(0.0, 0.0, 1.0));
        model = glm::rotate(model, (float)hpe_problem.get_model().get_pitch(), glm::vec3(0.0, 1.0, 0.0));
        model = glm::rotate(model, (float)hpe_problem.get_model().get_roll(), glm::vec3(1.0, 0.0, 0.0));

        modelShader.setMat4("projection", projection); // note: currently we set the projection matrix each frame, but since the projection matrix rarely changes it's often best practice to set it outside the main loop only once.
        modelShader.setMat4("view", view);    
        modelShader.setMat4("model", model);           

        glEnable(GL_DEPTH_TEST);
        glClear(GL_DEPTH_BUFFER_BIT);
        glBindVertexArray(faceVAO);
        glDrawElements(GL_TRIANGLES, 93322 * 3, GL_UNSIGNED_INT, 0);
        glDisable(GL_DEPTH_TEST);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
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
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}