#pragma once
#include <iostream>
#include <boost/asio.hpp>
#include <thread>

class Serial {
public:
    Serial(const std::string& portName) : port(io_context, portName) {
        init();
    }
    void send(const std::string& data);
    void receive() ;
private:
    void init() ;
    boost::asio::io_context io_context;
    boost::asio::serial_port port;
};

