#include <iostream>
#include <boost/asio.hpp>
#include <thread>
#include "Serial.h"


int main(int argc, char* argv[]) {
    if (argc!=2)
    {
        std::cout<<"Usage: "<<argv[0]<<" <serial_port>"<<std::endl;
        exit(-1);
    }
    
    Serial serial(argv[1]);

    std::thread sendThread([&]() {
        while (true) {
            std::string input;
            std::getline(std::cin, input);
            serial.send(input + "\n");
        }
    });

    std::thread receiveThread([&]() {
        serial.receive();
    });

    sendThread.join();
    receiveThread.join();

    return 0;
}
