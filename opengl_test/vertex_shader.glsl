#version 330 core

in vec4 position;
in vec2 texCoord;
out vec2 vColor;

void main()
{
	gl_Position = position;
	vColor = texCoord;
}
