#include <algorithm>
#include <iostream>
#include "data.hpp"

int main(int argc, char** argv){// main function that takes path for data
    std::string directory = (argc > 1) ? argv[1] : "data"; //if only 1 argument passed, assign 'data' to directory, if not use folder path passed to function

    try{
        //load training and test sets 
        FMNISTData train = load_fashion_mnist(directory + "/train-images-idx3-ubyte", directory + "/train-labels-idx1-ubyte"); 
        FMNISTData test = load_fashion_mnist(directory + "/t10k-images-idx3-ubyte", directory + "/t10k-labels-idx1-ubyte");
        //output size of training and testing sets 
        std::cout << "Training data: " << train.n << ", Test data: " << test.n << std::endl;

        //find the min and max of normalised pixel values to make sure normalisation was completed
        std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator> minmax = //place min and max values into a pair
            std::minmax_element(train.images[0].begin(), train.images[0].end()); //loop thorugh images (vector of pixel floats) to find min and max pixel values
        std::cout << "Min: " << *minmax.first << ", Max: " << *minmax.second << std::endl;//output min and max values

        std::vector<int> counts(10,0); //vector of length 10 filled with 0s, ready to hold counts of each label 
        for(std::size_t i = 0; i < train.labels.size(); ++i){ //loop through labels
            counts[train.labels[i]]++; //increment value for labels bucket when its encountered 
        }
        //output counts of each category 
        for (int c = 0; c < 10; ++c){
            std::cout << class_name(c) << ": " << counts[c] << std::endl;
        }

        const char* shades = " .:-=+*#%@"; //10 ASCII shades to represent different brightness pixels in order to output display of image 
        std::cout << "Label of image 0: " << class_name(train.labels[0]) << std::endl; //output name of first image in dataset
        for(int row = 0; row < train.rows; ++row){
            for(int col = 0; col < train.cols; ++col){
                char ch = shades[int(train.images[0][row * train.cols + col] * 9.99f)]; //convert pixel float to integer and use an index in string to find correct shade
                std::cout << ch << ch; //output shade to create image 
            }
            std::cout << std::endl;
        }
    }
    //catch any erros from operations in the try block
    catch(const std::exception& e){
        std::cerr << "error: " << e.what() << std::endl;
        return 1; //unsuccessful attempt at running main 
    }







}



