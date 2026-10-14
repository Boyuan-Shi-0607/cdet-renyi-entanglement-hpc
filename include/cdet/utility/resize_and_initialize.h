#ifndef RESIZE_AND_INITIALIZE_H
#define RESIZE_AND_INITIALIZE_H

#include <vector>
#include <complex>
#include <iostream>

// Base case for the recursion, where the vector is 1D
template <typename T>
void resizeVector(std::vector<T>& vec,
				  std::vector<int>::const_iterator shape_begin,
				  std::vector<int>::const_iterator shape_end) {
	if (shape_begin == shape_end) {
		// We've reached the innermost vector, initialize elements if needed
		return;
	}

	int size = *shape_begin;
	vec.resize(size, T(0));
}

// Recursive function template to handle multi-dimensional vectors
template <typename T>
void resizeVector(std::vector<std::vector<T>>& vec,
				  std::vector<int>::const_iterator shape_begin,
				  std::vector<int>::const_iterator shape_end) {
	if (shape_begin == shape_end) {
		// Should not happen; if shape is exhausted, we shouldn't have more vectors
		return;
	}

	int size = *shape_begin;
	vec.resize(size);  // Resize the outer vector
	for (auto& innerVec : vec) {
		resizeVector(innerVec, shape_begin + 1, shape_end);  // Recursively resize the inner vectors
	}
}

// Utility function to handle vector resizing based on shape with complex numbers
template <typename T>
void resizeMultiDimVector(std::vector<T>& vec, const std::vector<int>& shape) {
	if (shape.empty()) {
		throw std::invalid_argument("Shape cannot be empty");
	}
	resizeVector(vec, shape.begin(), shape.end());
}

#endif