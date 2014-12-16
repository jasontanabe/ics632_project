__kernel void stencil(__read_only image2d_t A, __write_only image2d_t B)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	int2 dims = get_image_dim(A);
	
	float sum = 0.0f;
	float4 pixels;
	if (i > 0) {
		pixels = read_imagef(A, (int2)(j, i-1));
		sum += 1.25*pixels.x;
	}
	if (i < dims.y) {
		pixels = read_imagef(A, (int2)(j, i+1));
		sum += pixels.x;
	}
	if (j > 0) {
		pixels = read_imagef(A, (int2)(j-1, i));
		sum += 1.25*pixels.x;
	}
	if (j < dims.x) {
		pixels = read_imagef(A, (int2)(j+1, i));
		sum += pixels.x;
	}

	sum /= 4.0f;
	
	if (sum > 1.0f) {
		sum = 1.0f;
	}
	
	pixels = (float4)(sum, 0.0f, 1.0f-sum, 1.0f);
	write_imagef(B, (int2)(j, i), pixels);
}