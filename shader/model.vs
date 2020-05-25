#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 color;

float fx = 1744.33;
float fy = 1747.84;
float cx = 800;
float cy = 600;
float width = 640;
float height = 480;

uniform mat4 model;
uniform bool isMirror;

void main()
{
	vec4 modelview = vec4(aPos, 1.0);
	vec3 ndc = modelview.xyz / modelview.w;
	float u = fx * ndc.x / ndc.z + cx;
	u = 2.0 * (u - width / 2) / width;
	float v = fy * ndc.y / ndc.z + cy;
	v = -2.0 * (v - height / 2) / height;
	float depth = ndc.z / 10000000;
	if(isMirror)
		gl_Position = vec4(-u, v, depth, 1.0);
	else
		gl_Position = vec4(u, v, depth, 1.0);
	color = aColor;
}
