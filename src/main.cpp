#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>



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
        std::string a = argv[i];
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
static Matrix make_batch(const FMNISTData& d, const vector<int>& idx, int start, int bs, vector<int>& y_out){
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



//find max value in row
static int argmax_row(const Matrix& m, int i){
    int best = 0; //set best value to first index by default
    for(int j = 1; j < m.cols; ++j){ //loop thorugh remaining columns
        if(m(i, j) > m(i, best)){
            best = j; //replace best with value found if its greater than current best
        }
    }
    return best; //return best value
}

//result evaluation struct
struct EvalResult{
    double loss;
    double accuracy;
};

//evaluate samples of the Dataset
static EvalResult evaluate(Network& net, const FMNISTData& d, int begin, int end){
    vector<int> idx; //index vector 
    for(int i = begin; i < end; ++i){
        idx.push_back(i); //add all indexes that will be used for evaulation
    }
    SoftmaxCrossEntropy ce;
    double loss = 0;
    int correct = 0;
    vector<int> y;
    for(int s = 0; s < static_cast<int>(idx.size()); s += 256){
        Matrix x = make_batch(d, idx, s, 256, y); //extract mini batch and labels
        Matrix out = net.forward(x); //forward pass
        loss += ce.forward(out, y) * out.rows; //total loss
        for(int i = 0; i < out.rows; ++i){
            correct += (argmax_row(out, i) == y[i]); //count number of true predicitons
        }
    }
    //add all results to evaluation result struct 
    EvalResult res;
    res.loss = loss/idx.size(); //mean loss
    res.accuracy = static_cast<double>(correct) / idx.size(); //classification accuracy ratio
    return res;
}

int main(int argc, char** argv){
    try{
        //create config object from args
        Config cfg = parse_args(argc, argv);

        //load training and testing data
        FMNISTData train = load_fashion_mnist(cfg.directory + "/train-images-idx3-ubyte", cfg.directory + "/train-image-idx1-ubyte");
        FMNISTData test = load_fashion_mnist(cfg.directory + "/t10k-images-idx3-ubyte", cfg.directory + "/t10k-image-idx1-ubyte");

        //validation set
        const int n_train = cfg.full ? train.n : train.n - 5000;

        //random seed for layer creation
        std::mt19937 rng(cfg.seed);

        //initialisers
        Init init = Init::He;
        if(cfg.initialiser == "naive"){
            init = Init::Naive;
        }else if(cfg.initialiser == "small"){
            init = Init::Naive;
        }

        //create network
        Network net;
        net.add(make_layer<Dense>(784, 128, rng, init));
        net.add(make_layer<ReLu>());
        net.add(make_layer<Dense>(128, 64, rng, init));
        net.add(make_layer<ReLu>());
        net.add(make_layer<Dense>(64, 10, rng, init));

        //optimisers
        std::unique_ptr<Optimiser> opt;
        opt.reset(new SGD(cfg.learning_rate, 0.0)); //default while Adam inst implemented
        if(cfg.optimiser == "sgd"){
            opt.reset(new SGD(cfg.learning_rate, 0.0));
        }else if(cfg.optimiser == "momentum"){
            opt.reset(new SGD(cfg.learning_rate, 0.9));
        }//else{ //Adam optimiser that hasn't been implemented yet
        //    opt.reset(new Adam(cfg.learning_rate));
        //}

        //check if csv is opened sucessfully and write header
        std::ofstream csv(cfg.out.c_str());
        if(!csv){
            throw std::runtime_error("Cannot write " + cfg.out);
        }
        csv << "epoch,train_loss,val_loss,val_acc,seconds \n";

        //SCE loss
        SoftmaxCrossEntropy loss;
        std::vector<int> idx(n_train);
        std::iota(idx.begin(), idx.end(), 0);
        std::vector<int> y;

        for(int e = 0; e < cfg.epochs; ++e){
            std::shuffle(idx.begin(), idx.end(), rng); //shuffle batch so program doesn't memorize batch order
            auto t0 = std::chrono::steady_clock::now();
            double total =0.0;
            int batches = 0;
            for(int s = 0; s < n_train; s += cfg.batch){ //loop through training set 
                Matrix x = make_batch(train, idx, s, cfg.batch, y); //creates mini batch
                total += loss.forward(net.forward(x), y); //compute predictions and calculate total loss
                net.backwards(loss.backwards());  //find loss gradient and backpropogate
                opt->step(net.params()); //applies optimiser to weights
                ++batches;
            }

            //second count of training pass
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); 

            //run evaluation to find current model loss and accuracy  
            EvalResult val = {0.0, 0.0};
            if(!cfg.full){
                val = evaluate(net, train, n_train, train.n);
            }

            //print summary for epoch
            std::cout << "epoch " << e << " train loss " << total / batches << " val loss " << val.loss << " val accuracy " << val.accuracy << " time:" << secs <<"s" << std::endl;
            //append row to CSV training log 
            csv << e << "," << total / batches << "," << val.loss << "," << val.accuracy << "," << secs << std::endl;
        }

        //evaluate trained network on test data
        EvalResult t = evaluate(net, test, 0, test.n);
        std::cout << "Test accuracy: " << t.accuracy << std::endl;

        //build 10x10 cm if output path is configured
        if(!cfg.confusion_matrix.empty()){
            std::vector<std::vector<int>> cm(
                10, std::vector<int>(10, 0)
            );

            //set samples to sequential order
            std::vector<int> all(test.n);
            std::iota(all.begin(), all.end(), 0);

            //run inference and count correct predictions and misclassifications
            for(int s = 0; s < test.n; s+= 256){
                Matrix x = make_batch(test, all, s, 256, y);
                Matrix out = net.forward(x);
                for(int i = 0; i < out.rows ; ++i){
                    cm[y[i]][argmax_row(out, i)]++;
                }
            }

            //write confusion matrix to CSV
            std::ofstream f(cfg.confusion_matrix.c_str());
            for(int r = 0; r < 10; ++r){
                for(int c = 0; c < 10; ++c){
                    f << cm[r][c] << (c < 9 ? "," : "\n");
                }
            }
        }
    }catch(const std::exception& ex){
        //report standard exceptions and exit
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
} 