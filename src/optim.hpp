#pragma once
#include <vector>
#include "layers.hpp"

struct Optimiser{
    virtual void step(const std::vector<Param*>& ps) = 0; //virtual interface for parameter updating function
    virtual ~Optimiser() {} //destructor
};

//
struct SGD : Optimiser{
    double learning_rate;
    double momentum;

    //constructor
    SGD(double lr, double m = 0.0)
        : learning_rate(lr), momentum(m) {}
    
        //step method to update trainable parameters
    void step(const std::vector<Param*>& ps) override{ //takes pointers to all parameters in the entire network
        for(std::size_t i = 0; i < ps.size(); ++i){
            Param* p = ps[i]; //store pointer to current parameter
            if(momentum > 0.0 && p->m.data.empty()){
                p->m = Matrix(p->value.rows, p->value.cols, 0.0); //create momentum velocity matrix
            }
            for(std::size_t j = 0; j < p->value.data.size(); ++j){
                //check if momentum is active
                if(momentum > 0.0){
                    p->m.data[j] = momentum * p->m.data[j] - learning_rate * p->grad.data[j]; //update velocity vector using momentum 
                    p->value.data[j] += p->m.data[j]; //accumulate velocity and update weight value
                }else{
                    p->value.data[j] -= learning_rate * p->grad.data[j]; //update weight value using gradient descent
                }
            }
        }
    }
};