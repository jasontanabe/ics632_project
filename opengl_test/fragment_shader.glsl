#version 330 core

in vec2 vColor;
out vec4 outColor;

uniform sampler2D texSampler;

void main()
{
	//outColor = vColor;
	outColor = texture(texSampler, vColor);
	//outColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}
