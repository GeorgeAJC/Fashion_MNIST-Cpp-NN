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
        //initialise Matricies with 0.0s 
        Weights.value = Matrix(in, out, 0.0);
        Weights.grad = Matrix(in, out, 0.0);
        Bias.value = Matrix(1, out, 0.0);
        Bias.grad = Matrix(1, out, 0.0);
        //standard deviation
        double stddev = std::sqrt(2.0 / in);
        
        //adjust standard deviations for intialisation methods
        if(init == Init::Naive){
            stddev = 1.0;
        }
        if(init == Init::Small){
            stddev = 0.01;
        }

        std::normal_distribution<double> nd(0.0, stddev);

        //randomly generated weights 
        for(std::size_t i = 0; i < Weights.value.data.size(); ++i){
            Weights.value.data[i] = nd(rng);
        }
    }
    //forward pass
    Matrix forward(const Matrix& input_matrix) override{
        input_cache = input_matrix;
        Matrix output_matrix = matmulAB(input_matrix, Weights.value);
        for(int i = 0; i < output_matrix.rows; ++i){
            for(int j = 0; j < output_matrix.cols; ++j){
                output_matrix(i, j) += Bias.value(0,j); // apply bias to matrix
            }
        }
        return output_matrix;
    }
    //backward pass 
    Matrix backward(const Matrix& grad_out) override{
        Weights.grad = matmulAtB(input_cache, grad_out); //calculate weight gradients
        for(int i = 0; i < grad_out.cols ; ++i){
            double total = 0.0;
            for(int j = 0; grad_out.rows; ++j){
                Bias.grad(0, i) = total; //calculate bias gradients 

            }
        }
        return matmulABt(grad_out, Weights.value); //retirn input gradients 
    }

    //return pointer to layers trainable parameters
    std::vector<Param*> params() override{
        std::vector<Param*> p;
        p.push_back(&Weights); //pass memory address of weight matrices
        p.push_back(&Bias); //pass memory addres of bias matriices
        return p;
    }


};

//Create Rectified Layer Unit Activation layer
struct ReLu : Layer{
    Matrix input_cache;


    //if number in matrix is less than 0.0, set it to 1, if its greater than 0.0 keep it the same (ReLu)
    Matrix forward(const Matrix& input_matrix) override{
        input_cache = input_matrix;
        Matrix output_matrix = input_matrix;
        for(std::size_t i = 0; i < output_matrix.data.size(); ++i){
            if(output_matrix.data[i] < 0.0){
                output_matrix.data[i] = 0.0;
            }
        }
        return output_matrix;
    }

    //if original input was less than or equal to 0.0 the derivative is 0, else it remains the same.
    Matrix backward(const Matrix& grad_out) override{
        Matrix output_matrix = grad_out;
        for(std::size_t i = 0; i < output_matrix.data.size(); ++i){
            if(input_cache.data[i] <= 0.0){
                output_matrix.data[i] = 0.0;
            }
        }
        return output_matrix;
    }


};

struct SoftmaxCrossEntropy{
    Matrix probabilities;
    std::vector<int> labels;

    //forwar pass for calculating softmax probabilities and sc cross entropy loss
    double forward(const Matrix& logits, const std::vector<int>& correct_labels){
        labels = correct_labels; //save correct labels for backwards pass
        probabilities = Matrix(logits.rows, logits.cols, 0.0);
        double loss = 0.0;
        for(int i = 0; i < logits.rows; ++i){    //find max logit
            double mx = logits(i, 0);
            for(int j = 1; j < logits.cols; ++j){
                mx = std::max(mx, logits(i, j));
            }
            //calculate exponention values and sum up
            double sum = 0.0;
            for(int j = 0; j < logits.cols; ++j){
                probabilities(i, j) = std::exp(logits(i,j) - mx); //subtract max to prevent floating point error
                sum += probabilities(i, j);
            }
            // normalise to [0,1] to get probabilities
            for(int j = 0; j < logits.cols; ++ j){
                probabilities(i, j) /= sum;
            }
            // add up negative logs of probabilities to get negative log-likelihood
            loss -= std::log(probabilities(i, correct_labels[i]) + 1e-12); //add small floating point number to prevent NaN values building up when a probability is equal to 0.0
        }
        return loss / logits.rows; //return mean loss
    }
    //backwards pass that computes gradients with respect to the logits
    Matrix backwards() const{
        Matrix grads = probabilities;
        for(int i = 0; i < grads.rows; ++i){ 
            grads(i, labels[i]) -= 1.0; //take 1 away from correct class (one-hot subtraction)
        }
        for(std::size_t j = 0; j < grads.data.size(); ++j){
            grads.data[j] /= grads.rows; //sacle gradient for backwards propagation
        }
        return grads; //return gradients 
    }
};

