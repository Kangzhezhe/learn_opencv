#include "Serial.h"


void Serial::send(const std::string& data) {
    try {
        boost::asio::write(port, boost::asio::buffer(data));
    } catch (const boost::system::system_error& e) {
        std::cerr << "Error sending data: " << e.what() << std::endl;
    }
}

void Serial::receive() {
    try {
        while (true) {
            std::string data;
            boost::asio::read_until(port, boost::asio::dynamic_buffer(data), '\n');
            std::cout << "Data received: " << data;
        }
    } catch (const boost::system::system_error& e) {
        std::cerr << "Error receiving data: " << e.what() << std::endl;
    }
}

void Serial::init() {
    try {
        port.set_option(boost::asio::serial_port_base::baud_rate(9600));
        port.set_option(boost::asio::serial_port_base::character_size(8));
        port.set_option(boost::asio::serial_port_base::parity(boost::asio::serial_port_base::parity::none));
        port.set_option(boost::asio::serial_port_base::stop_bits(boost::asio::serial_port_base::stop_bits::one));
        port.set_option(boost::asio::serial_port_base::flow_control(boost::asio::serial_port_base::flow_control::none));
    } catch (const boost::system::system_error& e) {
        std::cerr << "Error initializing serial port: " << e.what() << std::endl;
    }
}
