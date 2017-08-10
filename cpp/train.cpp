#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::SGDSolver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;
using std::ifstream;
using std::vector;
using std::cout;
using std::endl;

unsigned int be(const char *buf) {
  char buf2[4];
  buf2[0] = buf[3];
  buf2[1] = buf[2];
  buf2[2] = buf[1];
  buf2[3] = buf[0];
  return *(unsigned int *)buf2;
}

std::pair<vector<float>, int> load_image(const char *path) {
  char buf[1000];
  ifstream input(path, std::ios::binary);
  input.read(buf, 4);
  assert(be(buf) == 0x00000803);
  input.read(buf, 12);
  int number = be(buf);
  int rows   = be(buf+4);
  int cols   = be(buf+8);
  int total  = number * rows * cols;
  vector<float> alloc(total);
  std::transform(
    (std::istreambuf_iterator<char>(input)),
    std::istreambuf_iterator<char>(),
    alloc.begin(),
    [](unsigned char c) -> float {return ((float) c) / 255.f;});
  return std::make_pair(alloc, number);
}

 std::pair<vector<float>, int> load_label(const char *path) {
  char buf[10];
  ifstream input(path, std::ios::binary);
  input.read(buf, 4);
  assert(be(buf) == 0x00000801);
  input.read(buf, 4);
  int total = be(buf);
  vector<float> alloc(total);
  std::transform(
    (std::istreambuf_iterator<char>(input)),
    std::istreambuf_iterator<char>(),
    alloc.begin(),
    [](unsigned char c) -> float {return (float) c; });
  return std::make_pair(alloc, total);
}

const int ROUNDS = 199;
const int TESTBATCHSIZE = 64;
const int NUM_ITERS_PER_TEST = 2;
const int NUM_BATCHES_PER_ITER = 200;

int main2() {
  using std::ofstream;
  using std::ios;
  auto testim = load_image("model/t10k-images-idx3-ubyte");
  auto testlb = load_label("model/t10k-labels-idx1-ubyte");
  ofstream im;
  im.open("test-images.bin", ios::out | ios::binary);
  im.write((char*)testim.first.data(), testim.first.size()*sizeof(float));
  ofstream lb;
  lb.open("test-labels.bin", ios::out | ios::binary);
  lb.write((char*)testlb.first.data(), testlb.first.size()*sizeof(float));
}

int main() {

  Caffe::set_mode(Caffe::CPU);

  caffe::SignalHandler signal_handler(
        caffe::SolverAction::STOP,
        caffe::SolverAction::SNAPSHOT);

  auto images = load_image("model/train-images-idx3-ubyte");
  auto labels = load_label("model/train-labels-idx1-ubyte");
  auto testim = load_image("model/t10k-images-idx3-ubyte");
  auto testlb = load_label("model/t10k-labels-idx1-ubyte");
  assert(images.second == labels.second);

  caffe::SolverParameter solver_param1;
  caffe::ReadSolverParamsFromTextFileOrDie("model/mnist_quick_solver.prototxt", &solver_param1);
  shared_ptr<Solver<float> > 
    //  solver(new SGDSolver<float>(solver_param1));
     solver1(caffe::SolverRegistry<float>::CreateSolver(solver_param1));
  solver1->SetActionFunction(signal_handler.GetActionFunction());
 
  caffe::SolverParameter solver_param2;
  caffe::ReadSolverParamsFromTextFileOrDie("model/mnist_quick_solver.prototxt", &solver_param2);
  shared_ptr<Solver<float> > 
    //  solver(new SGDSolver<float>(solver_param1));
     solver2(caffe::SolverRegistry<float>::CreateSolver(solver_param2));
  solver2->SetActionFunction(signal_handler.GetActionFunction());

  auto solverNet = solver1->net();
  auto testerNet = solver2->net();
  testerNet->ShareTrainedLayersWith(solverNet.get());

  auto solver_memory_data = boost::static_pointer_cast<caffe::MemoryDataLayer<float> >(solverNet->layer_by_name("mnist"));
  solver_memory_data->Reset(images.first.data(), labels.first.data(), 60000/64*64);
  auto tester_memory_data = boost::static_pointer_cast<caffe::MemoryDataLayer<float> >(testerNet->layer_by_name("mnist"));
  tester_memory_data->Reset(testim.first.data(), testlb.first.data(), 10000/64*64);

  for (int i=0; i < ROUNDS; i++ ) {
    LOG(INFO) << "Iteration " << i;
    if (i % NUM_ITERS_PER_TEST == 0) {
      int iters = 10000 / TESTBATCHSIZE;
      float accuracy = 0;
      for (int j=0; j < iters; j++) {
        testerNet->Forward();
        accuracy += testerNet->blob_by_name("accuracy")->cpu_data()[0];
      }
      LOG(INFO) << "accuracy: " << accuracy / iters;
    }
    solver1->Step(NUM_BATCHES_PER_ITER);
  }

  LOG(INFO) << "Optimization Done.";
  return 0;
}
