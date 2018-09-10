#include <boost/thread.hpp>
#include <exception>

#include "caffe/segnet_internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SegInternalThread::~SegInternalThread() {
  StopInternalThread();
}

bool SegInternalThread::is_started() const {
  return thread_ && thread_->joinable();
}

bool SegInternalThread::must_stop() {
  return thread_ && thread_->interruption_requested();
}

void SegInternalThread::StartInternalThread() {
  CHECK(!is_started()) << "Threads should persist and not be restarted.";

  int device = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device));
#endif
  Caffe::Brew mode = Caffe::mode();
  int rand_seed = caffe_rng_rand();
  int solver_count = Caffe::solver_count();
  bool root_solver = Caffe::root_solver();

  try {
    thread_.reset(new boost::thread(&SegInternalThread::entry, this, device, mode,
          rand_seed, solver_count, root_solver));
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

void SegInternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
    int solver_count, bool root_solver) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device));
#endif
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);
  //Caffe::set_root_solver(root_solver);
  if(root_solver)
    Caffe::set_solver_rank(0);
  else
    Caffe::set_solver_rank(1);

  InternalThreadEntry();
}

void SegInternalThread::StopInternalThread() {
  if (is_started()) {
    thread_->interrupt();
    try {
      thread_->join();
    } catch (boost::thread_interrupted&) {
    } catch (std::exception& e) {
      LOG(FATAL) << "Thread exception: " << e.what();
    }
  }
}
/** Will not return until the internal thread has exited. */
bool SegInternalThread::WaitForInternalThreadToExit() {
  if (is_started()) {
    try {
      thread_->join();
    } catch (...) {
      return false;
    }
  }
  return true;
}

bool SegInternalThread::StartInternalThread2() {
  if (!WaitForInternalThreadToExit()) {
    return false;
  }
  try {
    thread_.reset(
        new boost::thread(&SegInternalThread::InternalThreadEntry, this));
  } catch (...) {
    return false;
  }
  return true;
}

}  // namespace caffe
