#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

 int main(int argc, char const *argv[])
 {
    if (argc<2){
        std::cout << " введите названия файла для проверки"<<std::endl;
    }
    std::ifstream file(argv[1]); // Открываем файл для чтения
    if (!file.is_open()) {
        std::cerr << "Unable to open file." << std::endl;
        return 1;
    }
    int all = 0;
    int accepted = 0;
    std::string line;
    while (std::getline(file, line)) { // Построчно считываем файл
        std::istringstream iss(line); // Создаем поток для текущей строки
        std::vector<std::string> tokens;

        std::string token;
        while (iss >> token) { // Разбиваем строку на токены
            tokens.push_back(token);
        }

        // Выводим полученные токены
        if (tokens[0]=="pow"){
            if (std::pow(stoi(tokens[1]),stoi(tokens[2]))==stoi(tokens[4])){
            accepted++;
            }
            all++;
        }
        //if (tokens[0]=="sin"){
            //if (std::abs(std::sin(stod(tokens[1]))==stod(tokens[3]))<1e-5){
                //accepted++;
           // }
        // почему была неправильная точность
          if (tokens[0]=="sin"){
              if (std::abs(std::sin(stod(tokens[1]))-stod(tokens[3]))<1e-5){
                 accepted++;
            }
           // std::cout<<stod(tokens[3])<<' '<< std::sin(stod(tokens[1])) << std::endl;
            all++;
        }
        if (tokens[0]=="sqrt"){
            if (std::abs(std::sqrt(stod(tokens[1]))-stod(tokens[3]))<1e-5){
                accepted++;
            }
           // std::cout<<stod(tokens[3])<<' '<< std::sqrt(stod(tokens[1])) << std::endl;
            all++;
        }
    }
    if (static_cast<double>(accepted)/static_cast<double>(all) > 0,9){
        std::cout<<"Passed"<<std::endl;
    }
    

    file.close(); // Закрываем файл

    return 0;
}
