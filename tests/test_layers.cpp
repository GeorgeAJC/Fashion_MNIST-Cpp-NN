#include <cmath>
#include<iostream>
#include "layers.hpp"

static int test_fails = 0;

int main(){
    //Testing for ReLu
    Matrix x(1, 3, 0.0);
    x(0, 0) = -1;
    x(0, 1) = 0;
    x(0, 2) = 2;
    ReLu relu;
    
    //test ReLu forward pass
    Matrix y = relu.forward(x);
    if(!(y(0,0) == 0 && y(0, 1) == 0 && y(0, 2) == 2)){
        std::cerr << "ReLu forward test failed" << std::endl;
        test_fails++;
    }

    //test relu backwards pass
    Matrix g(1, 3, 1.0);
    Matrix gi = relu.backwards(g);
    if (!(gi(0,0) == 0 && gi(0, 1) == 0 && gi(0, 2) == 1)){
        std::cerr << "ReLu backwards test failed" << std::endl;
        test_fails++;
    }

    //test SCE loss
    Matrix logits(4, 10, 0.0);
    std::vector<int> labels = {0, 1, 2, 3};
    SoftmaxCrossEntropy ce;
    double l = ce.forward(logits, labels);
    if (!(std::fabs(l- std::log(10.0)) < 1e-9)){
        std::cerr << "Failed SoftmaxCrossEntropy loss test" << std::endl;
        test_fails++;
    }

    //test SCE gradient summation
    Matrix gr = ce.backwards();
    for(int i = 0; i < gr.rows; ++i){
        double sum = 0.0;
        for(int j = 0; j < gr.cols; ++j){
            sum += gr(i, j);
        }
        if(!(std::fabs(sum) < 1e-12)){
            std::cerr << "Failed SoftmaxCrossEntropy gradient row sum test" << std::endl;
            test_fails++; 
        }
    }
    
    if (test_fails == 0){
        std::cout << "Tests passed"; //return success message if no tests failed 
        return 0;
    }
    std::cout << "Number of Tests Failed:" << test_fails; //return number of failed tests
    return 1;
}

