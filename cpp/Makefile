CAFFEDIR=../caffe/cppbuild/linux-x86_64
OPENCVDIR=../opencv/cppbuild/linux-x86_64
HDFDIR=../hdf5/cppbuild/linux-x86_64
OPENBLASDIR=../openblas/cppbuild/linux-x86_64
INCDIR=${CAFFEDIR}/include
LINKOPTS=-L${CAFFEDIR}/lib -L${OPENCVDIR}/lib -L${HDFDIR}/lib -Wl,-rpath-link ${HDFDIR}/lib -L${OPENBLASDIR}/lib -lcaffe -lboost_system -lboost_thread -lglog -lgflags -lleveldb -llmdb -lopenblas -lprotobuf -lsnappy -lboost_filesystem -pthread -lpthread -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lz
LDPATH=${CAFFEDIR}/lib:${OPENCVDIR}/lib:${HDFDIR}/lib:${OPENBLASDIR}/lib

train: train.cpp
	g++ -O3 -o train_mem -DCPU_ONLY -I${INCDIR} -std=gnu++11 -Wno-deprecated-declarations train_mem.cpp ${LINKOPTS}

run: train
	LD_LIBRARY_PATH=${LDPATH} ./train

dataset:
	/home/jwu/caffe/caffe/build/examples/mnist/convert_mnist_data.bin /home/jwu/caffe/caffe/data/mnist/train-images-idx3-ubyte /home/jwu/caffe/caffe/data/mnist/train-labels-idx1-ubyte model/mnist_train_lmdb --backend=lmdb
	/home/jwu/caffe/caffe/build/examples/mnist/convert_mnist_data.bin /home/jwu/caffe/caffe/data/mnist/t10k-images-idx3-ubyte /home/jwu/caffe/caffe/data/mnist/t10k-labels-idx1-ubyte model/mnist_test_lmdb --backend=lmdb
