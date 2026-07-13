// Originally derived from pbrt v4 (Apache 2.0)
// https://github.com/mmp/pbrt-v4
// Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
#pragma once

#include "CUDA/Annotations.h"

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace nr {

// TypePack + IndexOf
template <typename... Ts>
struct TypePack { static constexpr std::size_t count = sizeof...(Ts); };

template <typename T, typename Pack> struct IndexOf;

template <typename T, typename... Ts>
struct IndexOf<T, TypePack<T, Ts...>> { static constexpr int count = 0; };

template <typename T, typename U, typename... Ts>
struct IndexOf<T, TypePack<U, Ts...>> { static constexpr int count = 1 + IndexOf<T, TypePack<Ts...>>::count; };

namespace detail {

// Single recursive template instead of N hardcoded overloads.
// if constexpr terminates the chain at compile time; the compiler
// generates the same if/else ladder as before, with no runtime overhead.

template <typename F, typename R, typename T, typename... Rest>
NR_CPU_GPU R Dispatch(F&& f, const void* p, int i) {
    if (i == 0) return f(static_cast<const T*>(p));
    if constexpr (sizeof...(Rest) > 0)
        return Dispatch<F, R, Rest...>(std::forward<F>(f), p, i - 1);
    assert(false);
    return f(static_cast<const T*>(p));
}

template <typename F, typename R, typename T, typename... Rest>
NR_CPU_GPU R Dispatch(F&& f, void* p, int i) {
    if (i == 0) return f(static_cast<T*>(p));
    if constexpr (sizeof...(Rest) > 0)
        return Dispatch<F, R, Rest...>(std::forward<F>(f), p, i - 1);
    assert(false);
    return f(static_cast<T*>(p));
}

// CPU-only variants (no NR_CPU_GPU) — for DispatchCPU on the tagged pointer
template <typename F, typename R, typename T, typename... Rest>
R DispatchCPU(F&& f, const void* p, int i) {
    if (i == 0) return f(static_cast<const T*>(p));
    if constexpr (sizeof...(Rest) > 0)
        return DispatchCPU<F, R, Rest...>(std::forward<F>(f), p, i - 1);
    assert(false);
    return f(static_cast<const T*>(p));
}

template <typename F, typename R, typename T, typename... Rest>
R DispatchCPU(F&& f, void* p, int i) {
    if (i == 0) return f(static_cast<T*>(p));
    if constexpr (sizeof...(Rest) > 0)
        return DispatchCPU<F, R, Rest...>(std::forward<F>(f), p, i - 1);
    assert(false);
    return f(static_cast<T*>(p));
}

// Return type helpers
template <typename... Ts> struct IsSameType { static constexpr bool value = true; };
template <typename T>     struct IsSameType<T> { static constexpr bool value = true; };
template <typename T, typename U, typename... Ts>
struct IsSameType<T, U, Ts...> {
    static constexpr bool value = std::is_same_v<T, U> && IsSameType<U, Ts...>::value;
};

template <typename T, typename... Ts> struct SameType {
    using type = T;
    static_assert(IsSameType<T, Ts...>::value, "Dispatch return types must all be the same");
};

template <typename F, typename... Ts>
struct ReturnType { using type = typename SameType<std::invoke_result_t<F, Ts*>...>::type; };

template <typename F, typename... Ts>
struct ReturnTypeConst { using type = typename SameType<std::invoke_result_t<F, const Ts*>...>::type; };

template <typename F, typename R, typename Base, typename T, typename... Rest>
NR_CPU_GPU R DispatchObject(F&& f, Base* p, int i) {
    if (i == 0) return f(static_cast<T*>(p));
    if constexpr (sizeof...(Rest) > 0)
        return DispatchObject<F, R, Base, Rest...>(std::forward<F>(f), p, i - 1);
    assert(false);
    return f(static_cast<T*>(p));
}

template <typename F, typename R, typename Base, typename T, typename... Rest>
NR_CPU_GPU R DispatchObject(F&& f, const Base* p, int i) {
    if (i == 0) return f(static_cast<const T*>(p));
    if constexpr (sizeof...(Rest) > 0)
        return DispatchObject<F, R, Base, Rest...>(std::forward<F>(f), p, i - 1);
    assert(false);
    return f(static_cast<const T*>(p));
}

template <typename F, typename R, typename Base, typename T, typename... Rest>
R DispatchObjectCPU(F&& f, Base* p, int i) {
    if (i == 0) return f(static_cast<T*>(p));
    if constexpr (sizeof...(Rest) > 0)
        return DispatchObjectCPU<F, R, Base, Rest...>(std::forward<F>(f), p, i - 1);
    assert(false);
    return f(static_cast<T*>(p));
}

template <typename F, typename R, typename Base, typename T, typename... Rest>
R DispatchObjectCPU(F&& f, const Base* p, int i) {
    if (i == 0) return f(static_cast<const T*>(p));
    if constexpr (sizeof...(Rest) > 0)
        return DispatchObjectCPU<F, R, Base, Rest...>(std::forward<F>(f), p, i - 1);
    assert(false);
    return f(static_cast<const T*>(p));
}

} // namespace detail

// Type discriminator for polymorphic objects whose address is already owned
// elsewhere. Unlike TaggedPointer, this stores no self-referential pointer.
template <typename Base, typename... Ts>
class TaggedObject {
public:
    using Types = TypePack<Ts...>;

    TaggedObject() = default;

    template <typename T>
    NR_CPU_GPU TaggedObject(T* p) : tag_(p ? TypeIndex<T>() : 0) {}

    NR_CPU_GPU TaggedObject(std::nullptr_t) {}
    NR_CPU_GPU TaggedObject(const TaggedObject& other) : tag_(other.tag_) {}
    NR_CPU_GPU TaggedObject& operator=(const TaggedObject& other) {
        tag_ = other.tag_;
        return *this;
    }

    template <typename T>
    NR_CPU_GPU static constexpr unsigned int TypeIndex() {
        using Tp = std::remove_cv_t<T>;
        if constexpr (std::is_same_v<Tp, std::nullptr_t>) return 0;
        else return 1 + IndexOf<Tp, Types>::count;
    }

    NR_CPU_GPU unsigned int Tag() const { return tag_; }
    template <typename T> NR_CPU_GPU bool Is() const { return tag_ == TypeIndex<T>(); }
    NR_CPU_GPU static constexpr unsigned int MaxTag() { return sizeof...(Ts); }
    NR_CPU_GPU static constexpr unsigned int NumTags() { return MaxTag() + 1; }
    NR_CPU_GPU explicit operator bool() const { return tag_ != 0; }

    NR_CPU_GPU Base* ptr() {
        return tag_ != 0 ? static_cast<Base*>(this) : nullptr;
    }
    NR_CPU_GPU const Base* ptr() const {
        return tag_ != 0 ? static_cast<const Base*>(this) : nullptr;
    }

    template <typename T> NR_CPU_GPU T* Cast() {
        assert(Is<T>()); return static_cast<T*>(ptr());
    }
    template <typename T> NR_CPU_GPU const T* Cast() const {
        assert(Is<T>()); return static_cast<const T*>(ptr());
    }
    template <typename T> NR_CPU_GPU T* CastOrNullptr() {
        return Is<T>() ? static_cast<T*>(ptr()) : nullptr;
    }
    template <typename T> NR_CPU_GPU const T* CastOrNullptr() const {
        return Is<T>() ? static_cast<const T*>(ptr()) : nullptr;
    }

    template <typename F>
    NR_CPU_GPU decltype(auto) Dispatch(F&& f) {
        assert(ptr());
        using R = typename detail::ReturnType<F, Ts...>::type;
        return detail::DispatchObject<F, R, Base, Ts...>(
            std::forward<F>(f), ptr(), Tag() - 1);
    }

    template <typename F>
    NR_CPU_GPU decltype(auto) Dispatch(F&& f) const {
        assert(ptr());
        using R = typename detail::ReturnTypeConst<F, Ts...>::type;
        return detail::DispatchObject<F, R, Base, Ts...>(
            std::forward<F>(f), ptr(), Tag() - 1);
    }

    template <typename F>
    decltype(auto) DispatchCPU(F&& f) {
        assert(ptr());
        using R = typename detail::ReturnType<F, Ts...>::type;
        return detail::DispatchObjectCPU<F, R, Base, Ts...>(
            std::forward<F>(f), ptr(), Tag() - 1);
    }

    template <typename F>
    decltype(auto) DispatchCPU(F&& f) const {
        assert(ptr());
        using R = typename detail::ReturnTypeConst<F, Ts...>::type;
        return detail::DispatchObjectCPU<F, R, Base, Ts...>(
            std::forward<F>(f), ptr(), Tag() - 1);
    }

private:
    unsigned int tag_{};
};

// TaggedPointer — pointer + type tag packed into one uint64_t.
// Upper 7 bits = 1-based type index into Ts; lower 57 bits = address.
// Dispatches via recursive if constexpr chain instead of a vtable.
template <typename... Ts>
class TaggedPointer {
public:
    using Types = TypePack<Ts...>;

    TaggedPointer() = default;

    template <typename T>
    NR_CPU_GPU TaggedPointer(T* p) {
        uintptr_t iptr = reinterpret_cast<uintptr_t>(p);
        assert((iptr & ptrMask) == iptr);
        bits = iptr | (static_cast<uint64_t>(TypeIndex<T>()) << tagShift);
    }

    NR_CPU_GPU TaggedPointer(std::nullptr_t) {}
    NR_CPU_GPU TaggedPointer(const TaggedPointer& o) { bits = o.bits; }
    NR_CPU_GPU TaggedPointer& operator=(const TaggedPointer& o) { bits = o.bits; return *this; }

    template <typename T>
    NR_CPU_GPU static constexpr unsigned int TypeIndex() {
        using Tp = std::remove_cv_t<T>;
        if constexpr (std::is_same_v<Tp, std::nullptr_t>) return 0;
        else return 1 + IndexOf<Tp, Types>::count;
    }

    NR_CPU_GPU unsigned int Tag() const { return static_cast<unsigned int>((bits & tagMask) >> tagShift); }

    template <typename T> NR_CPU_GPU bool Is() const { return Tag() == TypeIndex<T>(); }

    NR_CPU_GPU static constexpr unsigned int MaxTag() { return sizeof...(Ts); }
    NR_CPU_GPU static constexpr unsigned int NumTags() { return MaxTag() + 1; }

    NR_CPU_GPU explicit operator bool() const { return (bits & ptrMask) != 0; }
    NR_CPU_GPU bool operator==(const TaggedPointer& o) const { return bits == o.bits; }
    NR_CPU_GPU bool operator!=(const TaggedPointer& o) const { return bits != o.bits; }
    NR_CPU_GPU bool operator< (const TaggedPointer& o) const { return bits <  o.bits; }

    template <typename T> NR_CPU_GPU T* Cast() {
        assert(Is<T>()); return reinterpret_cast<T*>(ptr());
    }
    template <typename T> NR_CPU_GPU const T* Cast() const {
        assert(Is<T>()); return reinterpret_cast<const T*>(ptr());
    }
    template <typename T> NR_CPU_GPU T* CastOrNullptr() {
        return Is<T>() ? reinterpret_cast<T*>(ptr()) : nullptr;
    }
    template <typename T> NR_CPU_GPU const T* CastOrNullptr() const {
        return Is<T>() ? reinterpret_cast<const T*>(ptr()) : nullptr;
    }

    NR_CPU_GPU void*       ptr()       { return reinterpret_cast<void*>(bits & ptrMask); }
    NR_CPU_GPU const void* ptr() const { return reinterpret_cast<const void*>(bits & ptrMask); }

    template <typename F>
    NR_CPU_GPU decltype(auto) Dispatch(F&& f) {
        assert(ptr());
        using R = typename detail::ReturnType<F, Ts...>::type;
        return detail::Dispatch<F, R, Ts...>(std::forward<F>(f), ptr(), Tag() - 1);
    }

    template <typename F>
    NR_CPU_GPU decltype(auto) Dispatch(F&& f) const {
        assert(ptr());
        using R = typename detail::ReturnTypeConst<F, Ts...>::type;
        return detail::Dispatch<F, R, Ts...>(std::forward<F>(f), ptr(), Tag() - 1);
    }

    template <typename F>
    decltype(auto) DispatchCPU(F&& f) {
        assert(ptr());
        using R = typename detail::ReturnType<F, Ts...>::type;
        return detail::DispatchCPU<F, R, Ts...>(std::forward<F>(f), ptr(), Tag() - 1);
    }

    template <typename F>
    decltype(auto) DispatchCPU(F&& f) const {
        assert(ptr());
        using R = typename detail::ReturnTypeConst<F, Ts...>::type;
        return detail::DispatchCPU<F, R, Ts...>(std::forward<F>(f), ptr(), Tag() - 1);
    }

private:
    static constexpr int      tagShift = 57;
    static constexpr int      tagBits  = 64 - tagShift;
    static constexpr uint64_t tagMask  = ((1ull << tagBits) - 1) << tagShift;
    static constexpr uint64_t ptrMask  = ~tagMask;
    uint64_t bits = 0;
};

} // namespace nr
