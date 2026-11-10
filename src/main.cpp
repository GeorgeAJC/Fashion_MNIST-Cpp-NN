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