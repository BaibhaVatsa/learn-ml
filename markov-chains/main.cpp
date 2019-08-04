#include <vector>
#include <iostream>

void printHelp();
std::vector<std::string> parseInput(int argc, char** argv);

int main(int argc, char** argv) {
    std::vector<std::string> authors = parseInput(argc, argv);

    return 0;
}

std::vector<std::string> parseInput(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "Usage: ./markov_chains --[author names]" << std::endl;
        std::cout << "./markov_chains --help for list of possible author names" << std::endl;
        exit(1);
    }

    std::vector<std::string> options;
    for(int i = 1; i <= argc; ++i) {
        options.emplace_back([&]() -> std::string {
            std::string option(argv[1]);
            if (option.length() > 2 && option[0] == '-' && option[1] == '-') {
                return option.substr(2);
            } else {
                std::cout << "Exiting. Error in parsing argument: " << option << "." << std::endl;
                exit(1);
            }
        }());
    }

    if(options[0] == "help") {
        printHelp();
        exit(1);
    }

    for(unsigned long i = 0; i < options.size(); ++i) {

    }

    return options;
}

void printHelp() {
    std::cout << "\t--all: ";

}