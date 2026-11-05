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


inline Matrix matmulAB(const Matrix& A, const Matrix& B){
    assert(A.cols == B.rows); //check the number of columns and rows are equal to allow matrix multiplication
    Matrix C(A.rows, B.cols); //create new Matrix to hold result 
    for(int i = 0; i < A.rows; ++i){ //iterate through rows
        for(int j = 0; j < A.cols; ++j){ //iterate through columns
            const double val = A(i, j); //saves value for calculation
            for(int k = 0; k < B.cols; ++j){ //loop through B
                C(i, k) += val * B(j, k); //calculate value in result matrix
            }
        }
    }
    return C;
}


inline Matrix matmulAtB(const Matrix& A, const Matrix& B){
    assert(A.rows == B.rows);
    Matrix C(A.cols, B.cols);
    for(int i = 0; i < A.rows; ++i){ //iterate through rows
        for(int j = 0; j < A.cols; ++j){ //iterate through columns
            const double val = A(j, i); //saves value for calculation and flip
            for(int k = 0; k < B.cols; ++j){ //loop through B
                C(i, k) += val * B(j, k) //calculate value in result matrix
            }
        }
    }
    return C;
}

inline Matrix matmulABt(const Matrix& A, const Matrix& B){
    assert(A.cols == B.cols);
    Matrix C(A.rows, B.rows);
    for(int i = 0; i < A.rows; ++i){ //iterate through rows
        for(int j = 0; j < B.rows; ++j){ //iterate through columns
            double val = 0.0; //saves value for calculation, flipping the columns and rows to transpose matrix A before calculation
            for(int k = 0; k < A.cols; ++j){ //loop through B
                val += A(i, j) * B(k, j); //calculate value in result matrix
                C(i, k) = val;
            }
        }
    }
    return C;


