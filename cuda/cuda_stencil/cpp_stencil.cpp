#include <iostream>
#include <vector>
#include <string>

#define N 600
#define ITERS 5

int main(int argc, char* argv[])
{
	std::vector<float> A(N*N), B(N*N);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A.at(i*N+j) = static_cast<float>(i+j);
//			std::cout << A.at(i*N+j) << " ";
		}

//		std::cout << std::endl;
	}
//	std::cout << std::endl;

	for (int num_it = 0; num_it < ITERS; num_it++) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				float update = 0.0;
				if (i > 0) {
					update += A.at((i-1)*N+j);
				}
				if (i < N-1) {
					update += A.at((i+1)*N+j);
				}
				if (j > 0) {
					update += A.at(i*N+(j-1));
				}
				if (j < N-1) {
					update += A.at(i*N+(j+1));
				}

				B.at(i*N+j) = update/4.0;
//				std::cout << B.at(i*N+j) << " ";
			}

//			std::cout << std::endl;
		}
		
//		std::cout << std::endl;
		A = B;
	}

  float sum = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum += B.at(i*N+j);
    }
  }
  std::cout << "sum is " << sum << std::endl;

	return 0;
}
