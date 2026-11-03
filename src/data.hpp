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

//function to load fashion MNIST data from files
inline FMNISTData load_fashion_mnist(const std::string& image_file, const std::string& label_file){ //inline to allow definition in header file and prevent multiple definition errors
    
    FMNISTData data;

    //validate that file exists and read image file
    std::ifstream file_input(image_file, std::ios::binary);
    if (!file_input.is_open()) {
        throw std::runtime_error("Could not open image file: " + image_file);
    }
    if (read_big_endian_uint32(file_input) != 2051){ //check magic number for image file
        throw std::runtime_error("Invalid image file format.");
    }
    //read number of images, rows and columns from image file 
    data.n = static_cast<int>(read_big_endian_uint32(file_input));
    data.rows = static_cast<int>(read_big_endian_uint32(file_input));   
    data.cols = static_cast<int>(read_big_endian_uint32(file_input));

    //read image data into a vector of unsigned char
    std::vector<unsigned char> raw(static_cast<size_t>(data.n) * data.rows * data.cols); //create a buffer to hold all raw image pixel data
    file_input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size())); //place raw data into buffer
    if (file_input.gcount() != static_cast<std::streamsize>(raw.size())) { //check all data was read
        throw std::runtime_error("Failed to read image data.");
    }

    //normalise pixel values and store each image as a vector of floats in the image vector
    const std::size_t pixels_per_image = static_cast<std::size_t>(data.rows) * data.cols;
    data.images.resize(data.n, std::vector<float>(pixels_per_image));
    for(std::size_t i = 0; i < raw.size(); ++i){ 
        data.images[i / pixels_per_image][i % pixels_per_image] 
            = static_cast<float>(raw[i]) / 255.0f; //normalise pixel values to [0,1]
    }

    //validate that label file exists and is of valid format and size
    std::ifstream file_labels(label_file, std::ios::binary);
    if (!file_labels.is_open()) {
        throw std::runtime_error("Can't open label file: " + label_file);
    }
    if (read_big_endian_uint32(file_labels) != 2049){
        throw std::runtime_error("Invalid label file format.");
    }
    if (static_cast<int>(read_big_endian_uint32(file_labels)) != data.n){
        throw std::runtime_error("Number of labels does not match number of images.");
    }

    //read raw label data into buffer and convert to target labels vector type 
    std::vector<unsigned char> raw_labels(static_cast<size_t>(data.n)); //create buffer for raw data
    file_labels.read(reinterpret_cast<char*>(raw_labels.data()), data.n); //read raw data into buffer
    if (file_labels.gcount() != data.n) {  //check if all data was read
        throw std::runtime_error("Failed to read label data.");
    }
    data.labels.assign(raw_labels.begin(), raw_labels.end()); //convert raw data to target labels vector type
    return data;

//return class name for each label
inline const char* class_name(int label){
    static const char* class_names[10] = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};
    return class_names[label];
}





}