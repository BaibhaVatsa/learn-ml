#include "Dictionary.h"

void Dictionary::addText(const std::string &filepath) {
//  given a file, add all its contents
    std::cout << filepath << std::endl;
    std::ifstream file(filepath);
    std::string prefix(DELIMITER), suffix("");
    while (file >> suffix) {
        std::cout << "prefix: " << prefix << " -> suffix: " << suffix << std::endl;
        chain[prefix].push_back(suffix);
        prefix = [&]() -> auto {
                    auto index = prefix.find_last_of(' ');
                    std::cout << index << std::endl;
                    return ((index != std::string::npos) ? prefix.substr(index+1) : "");
                 }() + " " + suffix;
    }
    std::cout << "prefix: " << prefix << " -> suffix: " << suffix << std::endl;
    chain[prefix].push_back(suffix);
}

void Dictionary::addAuthor(const std::string &filepath) {
//  for every file in the directory, read it and add it to the chain
    std::experimental::filesystem::path authorDirectory(filepath);
    for (const auto &file : std::experimental::filesystem::directory_iterator(authorDirectory)) {
        const auto &filename = file.path().filename().string();
        std::cout << "Adding " << filename << "..." << std::endl;
        addText(filepath+"/"+filename);
    }
}

void Dictionary::createText(const std::string &filepath, size_t size, size_t number){
//  create number documents of size words each in the filepath
    std::cout << filepath << std::endl;
    std::cout << size << number << std::endl;
}