#pragma once
#include <vector>

template <typename T>
class ObservableCollection {
public:
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    ObservableCollection() = default;

    // Access pointer for binding
    std::vector<T>* Ptr() { return &collection_; }
    const std::vector<T>* Ptr() const { return &collection_; }

    // Modifying methods
    void push_back(const T& value) {
        collection_.push_back(value);
        Notify();
    }
    void push_back(T&& value) {
        collection_.push_back(std::move(value));
        Notify();
    }
    iterator erase(const_iterator pos) {
        auto result = collection_.erase(pos);
        Notify();
        return result;
    }
    void clear() {
        collection_.clear();
        Notify();
    }

    // Access
    iterator begin() { return collection_.begin(); }
    const_iterator begin() const { return collection_.begin(); }
    iterator end() { return collection_.end(); }
    const_iterator end() const { return collection_.end(); }
    size_t size() const { return collection_.size(); }
    bool empty() const { return collection_.empty(); }
    T& operator[](size_t i) { return collection_[i]; }
    const T& operator[](size_t i) const { return collection_[i]; }

    // External trigger when bound value changes
    void Notify() {
        if (on_changed_)
            on_changed_();
    }

    void SetOnChanged(std::function<void()> callback) {
        on_changed_ = std::move(callback);
    }

private:
    std::vector<T> collection_;
    std::function<void()> on_changed_;
};
