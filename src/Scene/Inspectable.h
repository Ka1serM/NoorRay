#pragma once

#include <string>

class Inspectable {
public:
    virtual ~Inspectable() = default;

    virtual const std::string& getName() const = 0;
    virtual std::string getType() const = 0;
    virtual bool renderUi() = 0;
};
