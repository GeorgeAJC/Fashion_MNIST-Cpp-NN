#pragma once
#include <cstddef>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstddef>
#include <iostream>

//convert big endian to little endian and return as a integer
inline uint32_t read_big_endian_uint32(std::ifstream& file) {
    unsigned char bytes[4]; //array to hold the 4 bytes
    file.read(reinterpret_cast<char*>(bytes), 4); //read the bytes
    if (file.gcount() != 4) { //check if all 4 bytes were read
        throw std::runtime_error("Failed to read 4 bytes from file.");
    }
    //convert bytes into unit32_t in LE 
    return (uint32_t(bytes[0]) << 24) | 
        (uint32_t(bytes[1]) << 16) |
        (uint32_t(bytes[2]) << 8) |
        uint32_t(bytes[3]);
}

//structure to hold MNIST data
struct FMNISTData{
    std::vector<std::vector<float>> images; 
    std::vector<uint8_t> labels;     
    int n = 0; //number of images
    //image dimensions for 784 pixel image (28x28)
    int rows = 28;
    int cols = 28;
};