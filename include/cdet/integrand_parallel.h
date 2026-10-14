#ifndef BIPARTITE_LATTICE_PRINCIPAL_MINOR_CDET_INTEGRAND_PARALLEL_H
#define BIPARTITE_LATTICE_PRINCIPAL_MINOR_CDET_INTEGRAND_PARALLEL_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

constexpr size_t MAX_INTERNAL_PARALLEL_THREADS = 256;

inline int default_integrand_thread_count() {
    const unsigned detected = std::thread::hardware_concurrency();
    const unsigned bounded = detected == 0 ? 1U : std::min<unsigned>(detected, static_cast<unsigned>(MAX_INTERNAL_PARALLEL_THREADS));
    return static_cast<int>(bounded);
}

inline std::atomic<int>& integrand_thread_count_atomic() {
    static std::atomic<int> thread_count{1};
    return thread_count;
}

inline void set_integrand_internal_thread_count(int thread_count) {
    integrand_thread_count_atomic().store(std::max(1, thread_count), std::memory_order_relaxed);
}

inline int get_integrand_internal_thread_count() {
    return integrand_thread_count_atomic().load(std::memory_order_relaxed);
}

class IntegrandParallelExecutor {
public:
    using Task = std::function<void(size_t worker_index, size_t begin, size_t end)>;

    static IntegrandParallelExecutor& instance() {
        static IntegrandParallelExecutor executor;
        return executor;
    }

    size_t max_threads() const {
        return max_threads_;
    }

    void parallel_for(size_t requested_threads, size_t task_count, const Task& task) {
        const size_t worker_count = std::min(max_threads_, std::max<size_t>(1, requested_threads));
        if (worker_count <= 1 || task_count <= 1) {
            task(0, 0, task_count);
            return;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return !job_active_; });

        requested_worker_count_ = worker_count;
        task_count_ = task_count;
        task_ = &task;
        background_remaining_ = worker_count - 1;
        job_active_ = true;
        ++generation_;
        start_cv_.notify_all();
        lock.unlock();

        const auto [begin, end] = range_for_worker(0, worker_count, task_count);
        task(0, begin, end);

        lock.lock();
        done_cv_.wait(lock, [this] { return background_remaining_ == 0; });
        job_active_ = false;
        task_ = nullptr;
        lock.unlock();
        done_cv_.notify_all();
    }

private:
    IntegrandParallelExecutor()
        : max_threads_(std::max<size_t>(1, static_cast<size_t>(default_integrand_thread_count()))) {
        workers_.reserve(max_threads_ > 0 ? max_threads_ - 1 : 0);
        for (size_t worker_index = 1; worker_index < max_threads_; ++worker_index) {
            workers_.emplace_back([this, worker_index] { worker_loop(worker_index); });
        }
    }

    ~IntegrandParallelExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    static std::pair<size_t, size_t> range_for_worker(size_t worker_index,
                                                      size_t worker_count,
                                                      size_t task_count) {
        const size_t begin = (task_count * worker_index) / worker_count;
        const size_t end = (task_count * (worker_index + 1)) / worker_count;
        return {begin, end};
    }

    void worker_loop(size_t worker_index) {
        size_t seen_generation = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            start_cv_.wait(lock, [this, seen_generation] { return stop_ || generation_ != seen_generation; });
            if (stop_) {
                return;
            }

            seen_generation = generation_;
            const size_t worker_count = requested_worker_count_;
            const size_t task_count = task_count_;
            const Task* task = task_;

            if (worker_index >= worker_count || task == nullptr) {
                continue;
            }

            lock.unlock();
            const auto [begin, end] = range_for_worker(worker_index, worker_count, task_count);
            (*task)(worker_index, begin, end);
            lock.lock();

            if (background_remaining_ > 0) {
                --background_remaining_;
                if (background_remaining_ == 0) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    const size_t max_threads_;
    std::vector<std::thread> workers_;

    mutable std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;

    bool stop_ = false;
    bool job_active_ = false;
    size_t generation_ = 0;
    size_t requested_worker_count_ = 1;
    size_t task_count_ = 0;
    size_t background_remaining_ = 0;
    const Task* task_ = nullptr;
};

#endif
