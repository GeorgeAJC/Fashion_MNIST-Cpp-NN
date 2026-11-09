#pragma once
#include <vector>
#include "layers.hpp"

struct Optimiser{
    virtual void step(const std::vector<Param*>& ps) = 0; //template for parameter updating function
    virtual ~Optimiser() {} //destructor
};