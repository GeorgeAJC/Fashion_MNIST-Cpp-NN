#include <string>

#include "data.hpp"
#include "layers.hpp"
#include "optim.hpp"

using std::string;

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


