#pragma once
template <typename EventT>
class IObserver {
public:
    virtual ~IObserver() = default;
    // This method is called by the Observable when an event occurs
    virtual void OnNotified(const EventT& event) = 0;
};