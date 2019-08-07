#ifndef MARKOV_CHAINS_DICTIONARY_H
#define MARKOV_CHAINS_DICTIONARY_H

#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <experimental/filesystem>
#include <vector>
const std::string DELIMITER("|");

class Dictionary {
private:
    std::unordered_map<std::string, std::vector<std::string>> chain;
public:
//    Dictionary() = default;
    void addAuthor(const std::string &filepath);
    void createText(const std::string &filepath, size_t size, size_t number);
    void addText(const std::string &filepath);
};

#endif //MARKOV_CHAINS_DICTIONARY_H
