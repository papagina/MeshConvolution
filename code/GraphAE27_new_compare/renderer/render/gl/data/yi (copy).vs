#version 330 core
layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec3 a_Color;

out vec3 CamNormal;
out vec3 Color;

uniform mat4 ModelMat;
uniform mat4 PerspMat;

void main()
{
	gl_Position = PerspMat * ModelMat * vec4(a_Position, 1.0);
    CamNormal = (ModelMat * vec4(a_Normal, 0.0)).xyz;

    Color = a_Color;
}