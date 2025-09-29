#pragma once
#include <vector>
#include <algorithm>
#include "IObserver.h"

template <typename EventT>
class IObservable {
public:
    virtual ~IObservable() = default;
    virtual void AddObserver(IObserver<EventT>* o) = 0;
    virtual void RemoveObserver(IObserver<EventT>* o) = 0;
};


template <typename EventT>
class Observable : public IObservable<EventT> {
    std::vector<IObserver<EventT>*> observers;

public:
    // Implementation of IObservable methods
    void AddObserver(IObserver<EventT>* observer) override {
        observers.push_back(observer);
    }

    void RemoveObserver(IObserver<EventT>* observer) override {
        observers.erase(
            std::remove(observers.begin(), observers.end(), observer),
            observers.end()
        );
    }

protected:
    // Method used by concrete classes (like Scene) to trigger the update
    void Notify(const EventT& event) {
        auto copy = observers; // Copy to allow safe modification during iteration
        for (auto* o : copy)
            o->OnNotified(event);
    }
};