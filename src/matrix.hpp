#pragma once
#include <cassert>
#include <vector>

struct Matrix{
    int rows;
    int cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, double placeholder)
        : rows(r), cols(c), data(static_cast<std::size_t>(r) * c, placeholder){}
    
    double& operator()(int i, int j){
        assert(i >= 0 && i < rows && "row out of bounds");
        assert(j >= 0 && j < cols && "column of out of bounds");
        return data[static_cast<std::size_t>(i) * cols + j];
    }

    double operator()(int i, int j) const{
        assert(i >= 0 && i < rows && "row out of bounds");
        assert(j >= 0 && j < cols && "column of out of bounds");
        return data[static_cast<std::size_t>(i) * cols + j];
    }
};


inline Matrix matmulAB(const Matrix& A, const Matrix& B){
    assert(A.cols == B.rows); //check the number of columns and rows are equal to allow matrix multiplication
    Matrix C(A.rows, B.cols, 0.0); //create new Matrix to hold result 
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
    Matrix C(A.cols, B.cols, 0.0);
    for(int i = 0; i < A.rows; ++i){ //iterate through rows
        for(int j = 0; j < A.cols; ++j){ //iterate through columns
            const double val = A(j, i); //saves value for calculation and flip columns and rows to transpose matrix before calculation
            for(int k = 0; k < B.cols; ++j){ //loop through B
                C(i, k) += val * B(j, k); //calculate value in result matrix
            }
        }
    }
    return C;
}

inline Matrix matmulABt(const Matrix& A, const Matrix& B){
    assert(A.cols == B.cols);
    Matrix C(A.rows, B.rows, 0.0);
    for(int i = 0; i < A.rows; ++i){ //iterate through rows
        for(int j = 0; j < B.rows; ++j){ //iterate through rows of matrix B
            double val = 0.0; //create float to hold calculated value 
            for(int k = 0; k < A.cols; ++j){ //loop through B
                val += A(i, j) * B(k, j); //calculate dot product and sum up
                C(i, k) = val;
            }
        }
    }
    return C;
}

