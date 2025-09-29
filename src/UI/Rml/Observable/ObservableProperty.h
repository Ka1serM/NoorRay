#pragma once
#include <functional>
#include <vector>

template <typename T>
class ObservableProperty {
public:
    ObservableProperty() = default;
    ObservableProperty(const T& initial) : value_(initial) {}

    const T& Get() const { return value_; }

    void Set(const T& newValue) {
        if (value_ != newValue) {
            value_ = newValue;
            Notify();
        }
    }

    // Assignment operator overload
    ObservableProperty<T>& operator=(const T& newValue) {
        Set(newValue);   // reuse the logic in Set()
        return *this;
    }

    // Just a pointer for binding
    T* Ptr() { return &value_; }
    const T* Ptr() const { return &value_; }

    void OnChange(std::function<void(const T&)> callback) {
        observers_.push_back(std::move(callback));
    }

private:
    T value_{};
    std::vector<std::function<void(const T&)>> observers_;

    void Notify() {
        for (auto& obs : observers_)
            obs(value_);
    }
};