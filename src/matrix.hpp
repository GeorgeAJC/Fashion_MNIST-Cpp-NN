#pragma once
#include <assert>
#include <vector>

struct Matrix{
    int rows;
    int cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, double placeholder)
        : rows(r), cols(c), data(static_cast<std::size_t>(r) * c, placeholder){}
    
    double& operator()(int i, int j){
        return data[static_cast<std::size_t>(i) * cols + j];
    }

    double operator()(int i, int j) const{
        return data[static_cast<std::size_t>(i) * cols + j];
    }
};


inline Matrix matmul(const Matrix& A, const Matrix& B){
    assert(A.cols == B.rows); //check the number of columns and rows are equal to allow matrix multiplication
    Matrix C(A.rows, B.cols); //create new Matrix to hold result 
    for(int i = 0; i < A.rows; ++i){ //iterate through rows
        for(int j = 0; j < A.cols; ++j){ //iterate through columns
            const double a = A(i, j); //saves value for calculation
            for(int k = 0; k < B.cols; ++j){ //loop through B
                C(i, k) += a * B(j, k) //calculate value in result matrix
            }
        }
    }
    return C;
}
 