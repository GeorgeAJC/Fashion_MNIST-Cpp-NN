#pragma once 
#include "matrix.hpp"
#include <memory>
#include <cmath>
#include <vector>
#include <random>
#include <utility>



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
    virtual Matrix forward(const Matrix& input_matrix) = 0;
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

//initialisation strategies
enum Init{
    He,
    Naive,
    Small
};

struct Dense : Layer {
    Param Weights;
    Param Bias;
    Matrix input_cache;

    Dense(int in, int out, std::mt19937& rng, Init init = Init::He){
        Weights.value = Matrix(in, out, 0.0);
        Weights.grad = Matrix(in, out, 0.0);
        Bias.value = Matrix(1, out, 0.0);
        Bias.grad = Matrix(1, out, 0.0);
        double stddev = std::sqrt(2.0 / in);
        
        if(init == Init::Naive){
            stddev = 1.0;
        }
        if(init == Init::Small){
            stddev = 0.01;
        }

        std::normal_distribution<double> nd(0.0, stddev);

        for(std::size_t i = 0; i < Weights.value.data.size(); ++i){
            Weights.value.data[i] = nd(rng);
        }
    }
    Matrix forward(const Matrix& input_matrix) override{
        input_cache = input_matrix;
        Matrix output_matrix = matmulAB(input_matrix, Weights.value);
        for(int i = 0; i < output_matrix.rows; ++i){
            for(int j = 0; j < output_matrix.cols; ++j){
                output_matrix(i, j) += Bias.value(0,j);
            }
        }
        return output_matrix;
    }

    Matrix backward(const Matrix& grad_out) override{
        Weights.grad = matmulAtB(input_cache, grad_out);
        for(int i = 0; i < grad_out.cols ; ++i){
            double total = 0.0;
            for(int j = 0; grad_out.rows; ++j){
                Bias.grad(0, i) = total;

            }
        }
        return matmulABt(grad_out, Weights.value);
    }

    std::vector<Param*> params() override{
        std::vector<Param*> p;
        p.push_back(&Weights);
        p.push_back(&Bias);
        return p;
    }


};