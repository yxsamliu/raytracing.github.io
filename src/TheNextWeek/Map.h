#pragma once

#include "List.h"
#include "Vector.h"
#include <stdexcept>
#include <utility>

template <typename T1, typename T2> class Pair {
public:
  T1 first;
  T2 second;

  // Constructor
  __host__ __device__ Pair(const T1 &a, const T2 &b) : first(a), second(b) {}

  // Default constructor
  __host__ __device__ Pair() : first(T1()), second(T2()) {}
};

template <typename KeyType, typename ValueType> class Map {
private:
  size_t bucketCount;
  Vector<List<Pair<KeyType, ValueType>>> buckets;

  __host__ __device__ size_t getBucketIndex(const KeyType &key) const {
    return std::hash<KeyType>{}(key) % bucketCount;
  }

public:
  __host__ __device__ Map(size_t numBuckets = 11)
      : bucketCount(numBuckets), buckets(numBuckets) {}

  __host__ __device__ Map(const Map &other)
      : buckets(other.buckets), bucketCount(other.bucketCount) {}

  __host__ __device__ Map &operator=(const Map &other) {
    if (this != &other) {
      buckets = other.buckets;
      bucketCount = other.bucketCount;
    }
    return *this;
  }

  __host__ __device__ void set(const KeyType &key, const ValueType &value) {
    size_t bucketIndex = getBucketIndex(key);
    auto &bucket = buckets[bucketIndex];

    for (auto &pair : bucket) {
      if (pair.first == key) {
        pair.second = value;
        return;
      }
    }

    bucket.emplace_back(key, value);
  }

  __host__ __device__ bool get(const KeyType &key, ValueType &value) const {
    size_t bucketIndex = getBucketIndex(key);
    const auto &bucket = buckets[bucketIndex];

    for (const auto &pair : bucket) {
      if (pair.first == key) {
        value = pair.second;
        return true;
      }
    }

    return false;
  }

  __host__ __device__ bool remove(const KeyType &key) {
    size_t bucketIndex = getBucketIndex(key);
    auto &bucket = buckets[bucketIndex];

    for (auto it = bucket.begin(); it != bucket.end(); ++it) {
      if (it->first == key) {
        bucket.erase(it);
        return true;
      }
    }

    return false;
  }

  __host__ __device__ ValueType &operator[](const KeyType &key) {
    size_t bucketIndex = getBucketIndex(key);
    auto &bucket = buckets[bucketIndex];

    for (auto &pair : bucket) {
      if (pair.first == key) {
        return pair.second;
      }
    }

    // Key not found, create a new element with default-initialized value
    bucket.emplace_back(key, ValueType{});
    return bucket.back().second;
  }

  // Custom iterator class
  class Iterator {
  private:
    typename Vector<List<Pair<KeyType, ValueType>>>::iterator bucketIt;
    typename Vector<List<Pair<KeyType, ValueType>>>::iterator bucketItEnd;
    typename List<Pair<KeyType, ValueType>>::iterator listIt;

  public:
    __host__ __device__
    Iterator(typename Vector<List<Pair<KeyType, ValueType>>>::iterator start,
             typename Vector<List<Pair<KeyType, ValueType>>>::iterator end)
        : bucketIt(start), bucketItEnd(end) {
      if (bucketIt != bucketItEnd) {
        listIt = bucketIt->begin();
        advancePastEmptyBuckets();
      }
    }

    __host__ __device__ void advancePastEmptyBuckets() {
      while (bucketIt != bucketItEnd && listIt == bucketIt->end()) {
        ++bucketIt;
        if (bucketIt != bucketItEnd)
          listIt = bucketIt->begin();
      }
    }

    __host__ __device__ Iterator &operator++() {
      ++listIt;
      advancePastEmptyBuckets();
      return *this;
    }

    __host__ __device__ Pair<KeyType, ValueType> *operator->() {
      return &(*listIt);
    }

    __host__ __device__ Pair<KeyType, ValueType> &operator*() {
      return *listIt;
    }

    __host__ __device__ bool operator==(const Iterator &other) const {
      return bucketIt == other.bucketIt &&
             (bucketIt == bucketItEnd || listIt == other.listIt);
    }

    __host__ __device__ bool operator!=(const Iterator &other) const {
      return !(*this == other);
    }
  };

  __host__ __device__ Iterator begin() {
    return Iterator(buckets.begin(), buckets.end());
  }

  __host__ __device__ Iterator end() {
    return Iterator(buckets.end(), buckets.end());
  }

  __host__ __device__ Iterator find(const KeyType &key) {
    size_t bucketIndex = getBucketIndex(key);
    auto &bucket = buckets[bucketIndex];
    for (auto it = bucket.begin(); it != bucket.end(); ++it) {
      if (it->first == key) {
        return Iterator(buckets.begin() + bucketIndex, buckets.end());
      }
    }
    return end();
  }
};
