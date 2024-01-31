#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>
#include <fstream>

namespace supertensor
{
    namespace cputucker
    {

        class Config
        {
        public:
            Config(const std::string &config_path)
            {
                boost::property_tree::read_json(config_path, config);
            }

            std::string getFilePath(const std::string &key)
            {
                return config.get<std::string>("file_paths." + key);
            }

        private:
            boost::property_tree::ptree config;
        };
    }
}

#endif // CONFIG_HPP