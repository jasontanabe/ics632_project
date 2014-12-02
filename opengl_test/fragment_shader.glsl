#version 330 core

in vec2 vColor;
out vec4 outColor;

uniform sampler2D texSampler;

void main()
{
	outColor = texture(texSampler, vColor);
}
