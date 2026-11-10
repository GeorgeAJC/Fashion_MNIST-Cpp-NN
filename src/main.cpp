#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>



#include "data.hpp"
#include "layers.hpp"
#include "optim.hpp"

using std::string;
using std::vector;

struct Config{
    string directory = "data";
    string optimiser = "adam";
    string initialiser = "he";
    string out = "results/run.csv";
    string confusion_matrix  = "";
    int epochs = 20;
    int batch = 64;
    double learning_rate = 0.001;
    unsigned seed = 0;
    bool full = false;
};

//helper function to popuulate config structs
static Config parse_args(int argc, char** argv){
    Config c;
    //loop through each arguement
    for(int i = 1; i < argc; ++i){
        std::string a = argc[i];
        if(a == "--full"){
            c.full = true;
            continue;
        }
        //check for missing flags, terminate if found
        if(i+1 >= argc){
            std::cerr << "missing value for" << a << std::endl;
            std::exit(1) ;
        }
        std::string v = argv[++i]; //store arg value
        //build config struct using arg values
        if(a == "--data") c.directory = v;
        else if(a == "--opt") c.optimiser = v;
        else if(a == "--init") c.initialiser = v;
        else if(a == "--out") c.out = v;
        else if(a == "--cm") c.confusion_matrix = v;
        else if(a == "--epochs") c.epochs = std::stoi(v);
        else if(a == "--batch") c.batch = std::stoi(v);
        else if(a == "--lr") c.learning_rate = std::stoi(v);
        else if(a == "--seed") c.seed = static_cast<unsigned>(std::stoul(v));
        //if unrecognised flag parsed then terminate 
        else{
            std::cerr << "unknown flag" << a << std::endl;
            std::exit(1);
        }
    }
    return c; //return config struct
}

//helper functions to from minibatches of image+labels for training and evaluation
static Matrix make_batch(const FMNISTData& d, const vector<int>& idx, int start, int bs, vector<int>& y_out{
    const int n = std::min (bs, static_cast<int>(idx.size()) - start); //batch size with last batch being the size of the amount of data left in the dataset
    const int dimensions = d.rows * d.cols; //dimensions of each image
    Matrix x(n, dimensions, 0.0); //create a batch matrix
    y_out.resize(n); //make label vector the size of images in the batch
    for(int i = 0; i < n; ++i){
        const int s = idx[start + i]; //index of image for the row
        const float* src = d.images[s].data(); //pointer for first image pixel 
        for(int j = 0; j < dimensions; ++j){
            x(i, j) = src[j]; //copy pixel
        }
        y_out[i] = d.labels[s]; //store label of image
    }
    return x; //return batch 
}