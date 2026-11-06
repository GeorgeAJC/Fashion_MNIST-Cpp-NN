#pragma once 
#include "matrix.hpp"
#include <memory>
#include <vector>


struct Param{
    Matrix value;
    Matrix grad;
    //created for future Adam optimiser
    Matrix m;
    Matrix v;
};

//layers for network
struct Layer{
    virtual ~Layer() {}
    virtual Matrix forward(const Matrix& x) = 0;
    virtual Matrix backward(const Matrix& grad_out) = 0;
    virtual std::vector<Param*> params(){
        return std::vector<Param*>();
    }
};

//unique pointers for layer struct 
template <typename T, typename... Args>
std::unique_ptr<Layer> make_layer(Args&&... args){
    return std::unique_ptr<Layer>(new T(std::forward<Args>(args)...));
}

enum Init{
    He,
    Naive,
    Small
};