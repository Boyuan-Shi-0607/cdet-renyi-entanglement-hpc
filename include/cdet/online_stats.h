#ifndef ONLINE_STATS_H
#define ONLINE_STATS_H

#include <iostream>
#include <vector>
#include <numeric>
#include "cdet/utility/types.h"
#include "cdet/utility/my_funcs.h"
#include <cmath>

template<typename T>
class OnlineStatsT {
public:
    OnlineStatsT(int M = 8) : count(0), mean(T(0)), M2(T(0)), maxBlocks(M) {
        // Initialize the blocks vector with M empty vectors
        blocks.resize(maxBlocks);

        // Initialize temporary storage for chunk means
        chunkCounts.resize(maxBlocks, 0);
        chunkSums.resize(maxBlocks, T(0));
    }

    // Update the running statistics with a new sample x.
    void update(T x) {
        ++count;
        T delta = x - mean;
        mean = mean + delta / T(count);
        T delta2 = x - mean;
        M2 = M2 + delta * delta2;

        // Update chunk statistics for each block size
        for (int i = 0; i < maxBlocks; ++i) {
            // Update this chunk's statistics
            chunkSums[i] = chunkSums[i] + x;
            chunkCounts[i]++;

            // If we've completed a chunk, record its mean and reset the chunk statistics
            if ((count - 1) % (1024 * (1 << i)) == (1024 * (1 << i)) - 1) {
                blocks[i].push_back(chunkSums[i] / T(chunkCounts[i]));
                chunkSums[i] = T(0);
                chunkCounts[i] = 0;
            }
        }
    }

    // Return the number of samples processed.
    int getCount() const { return count; }

    // Return the current running mean.
    T getMean() const { return mean; }

    // Return the sample variance (with Bessel's correction).
    // If fewer than 2 samples have been processed, return 0.
    T getVariance() const {
        return (count > 1) ? M2 / T(count - 1) : T(0);
    }

    T getStdDev() const {
        T var = getVariance();
        if constexpr (std::is_same_v<T, DoubleDouble>) {
            return DD_SQRT(var / T(count));
        } else {
            return std::sqrt(var / count);
        }
    }

    // Get the vector of means recorded every blockSize samples
    const std::vector<T>& getBlockMeans(int blockIndex) const {
        if (blockIndex < 0 || blockIndex >= maxBlocks) {
            throw std::out_of_range("Block index out of range");
        }
        return blocks[blockIndex];
    }

    // Get the block size for a given block index
    int getBlockSize(int blockIndex) const {
        if (blockIndex < 0 || blockIndex >= maxBlocks) {
            throw std::out_of_range("Block index out of range");
        }
        return 1024 * (1 << blockIndex);
    }

    // Get the maximum number of blocks
    int getMaxBlocks() const {
        return maxBlocks;
    }

    // Public members
    int count;
    T mean;    // Running mean.
    T M2;      // Sum of squares of differences from the mean.
    int maxBlocks;  // Maximum number of different block sizes
    std::vector<std::vector<T>> blocks;  // Vector of vectors storing means at different intervals

    // Temporary storage for calculating chunk means
    std::vector<int> chunkCounts;  // Count of samples in current chunk for each block size
    std::vector<T> chunkSums;   // Sum of samples in current chunk for each block size
};

// Legacy typedef for backward compatibility
using OnlineStats = OnlineStatsT<Real>;

#endif //ONLINE_STATS_H