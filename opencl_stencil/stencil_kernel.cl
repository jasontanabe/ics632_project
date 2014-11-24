__kernel void stencil(__global const float *A, __global float *B, __global const int *N) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	float sum = 0.0;
	int lN = *N;
	if (i > 0) {
		sum += A[(i-1)*lN+j];
	}
	if (i < lN-1) {
		sum += A[(i+1)*lN+j];
	}
	if (j > 0) {
		sum += A[i*lN+(j-1)];
	}
	if (j < lN-1) {
		sum += A[i*lN+(j+1)];
	}

	B[i*lN+j] = sum / 4.0;
}
