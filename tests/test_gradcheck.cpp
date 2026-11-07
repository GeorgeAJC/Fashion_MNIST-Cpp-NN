#include <algorithm>
#include <cmath>
#include <iostream>
#include "layers.hpp"

int main(){
    std::mt19937 rng(42); //fixed seed random number generator
    Network net; //create network container
    //build small test model
    net.add(make_layer<Dense>(20, 16, rng)); //Dense layer of 20 inputs and 16 hidden nodes
    net.add(make_layer<ReLu>()); //ReLu activation layer
    net.add(make_layer<Dense>(16, 10, rng)); //Dense layer of 16 inputs and 10 hidden nodes

    Matrix x(8, 20, 0.0); //test matrix
    std::normal_distribution<double> nd(0.0, 1.0);

    //fill matrix with random normalised test values between 0 and 1
    for(std::size_t i = 0; i < x.data.size(); ++i){
        x.data[i] = nd(rng); 
    }
    std::vector<int> y = {0,1,2,3,4,5,6,7}; //create labels 

    SoftmaxCrossEntropy loss;
    //create lambda to compute SCE loss
    auto compute_loss = [&](){
         return loss.forward(net.forward(x), y);
    };

    //initial forward pass and gradient computation
    compute_loss();
    net.backwards(loss.backwards());

    const double eps = 1e-5; //pertubation step size for differentitation
    double worst = 0.0; //worst relative error
    std::vector<Param*> ps = net.params(); //pointers to each all trainable weight and bias matrices
    for(std::size_t p = 0; p < ps.size(); ++p){
        Param* prm = ps[p]; //retrieve pointer to current parameter
        for(std::size_t i = 0; i < prm->value.data.size(); i += 5){ //sample every 5th wait for faster testing
            //save original weight value
            const double orig = prm->value.data[i];
            //calculate positive pertubed loss
            prm->value.data[i] = orig + eps;
            const double lp = compute_loss();
            //calculate negative pertubed loss
            prm->value.data[i] = orig - eps;
            const double lm = compute_loss();
            //set weight back to original value. 
            prm->value.data[i] = orig;
            
            //compute gradient
            const double num = (lp-lm) / (2*eps);
            //compute analytical gradient 
            const double ana = prm->grad.data[i];
            //create denomiator for division when gradient is close to zero
            const double denom = std::max<double>(1e-8, std::fabs(num) + std::fabs(ana));
            //find worst relative error 
            worst = std::max(worst, std::fabs(num - ana) / denom);
        }
    }

    std::cout << "worst relative error: " << worst << std::endl; //print relative error
    if(worst < 1e-5){
        std::cout << "test passed" << std::endl;
        return 0; //success if relative error is less than 1e-5
    }
    std::cout << "test failed" << std::endl;
    return 1; //fail if relative error is more than 1e-5



}