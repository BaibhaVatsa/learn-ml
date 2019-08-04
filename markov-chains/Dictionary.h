#ifndef MARKOV_CHAINS_DICTIONARY_H
#define MARKOV_CHAINS_DICTIONARY_H

#include <string>

class Dictionary {
private:
    std::string author;
public:
    explicit Dictionary(const std::string &auth);
};

#endif //MARKOV_CHAINS_DICTIONARY_H
