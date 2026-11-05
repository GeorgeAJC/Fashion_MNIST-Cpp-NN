#pragma once
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


