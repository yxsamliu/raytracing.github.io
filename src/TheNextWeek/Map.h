#pragma once

#include <list>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename KeyType, typename ValueType> class Map {
private:
  size_t bucketCount;
  std::vector<std::list<std::pair<KeyType, ValueType>>> buckets;

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
    typename std::vector<std::list<std::pair<KeyType, ValueType>>>::iterator
        bucketIt;
    typename std::vector<std::list<std::pair<KeyType, ValueType>>>::iterator
        bucketItEnd;
    typename std::list<std::pair<KeyType, ValueType>>::iterator listIt;

  public:
    __host__ __device__ Iterator(
        typename std::vector<std::list<std::pair<KeyType, ValueType>>>::iterator
            start,
        typename std::vector<std::list<std::pair<KeyType, ValueType>>>::iterator
            end)
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

    __host__ __device__ std::pair<KeyType, ValueType> *operator->() {
      return &(*listIt);
    }

    __host__ __device__ std::pair<KeyType, ValueType> &operator*() {
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
