import org.apache.spark.sql.types._
import org.apache.spark.sql.SparkSession
import org.bytedeco.javacpp.caffe._
import org.bytedeco.javacpp.{Pointer, IntPointer, PointerPointer, FloatPointer}

//
// Solver wraps a lazy creation of CaffeSolver. In this way, a solver
// instance can be serialized and the wrapped CaffeSolver will be
// created on each Spark executor.
//
class Solver(val args: Array[String], val home: String) extends Serializable {
  // keep netParam and solverParam as a member here, so that they won't be GC'ed.
  lazy val netParam = new NetParameter()
  lazy val solverParam = new SolverParameter()
  lazy val instance = {
    // import org.bytedeco.javacpp.Loader
    // import org.bytedeco.javacpp.caffe
    // Loader.load(classOf[caffe])
    // val pargc = new IntPointer(Array(1) :_*)
    // val argv  = new PointerPointer(Array("mnist") : _*)
    // val pargv = new PointerPointer(Array(argv) : _*)
    //GlobalInit(pargc, pargv)
 
    ReadProtoFromTextFileOrDie(home + "/model/mnist_net.prototxt", netParam)
    ReadSolverParamsFromTextFileOrDie(home + "/model/mnist_solver.prototxt", solverParam)
    solverParam.clear_net()
    solverParam.set_allocated_net_param(netParam)
    Caffe.set_mode(Caffe.CPU)
    val solver = new CaffeSolver(solverParam)
    solver
  }
}

object MnistApp {
  val trainBatchSize = 64
  val testBatchSize  = 64
  val num_batches_per_iter = 5
  val num_iters_per_round = 20

  def main(args: Array[String]) {

    val mnistHome   = sys.env("MNIST_HOME")
    val numWorkers = args(0).toInt

    val spark = SparkSession
      .builder()
      .appName("Mnist")
      .getOrCreate()
    import spark.implicits._
    val sc = spark.sparkContext

    val logger = new Logger(mnistHome + "/training_log_" + System.currentTimeMillis().toString + ".txt")
    val loader = new Loader(mnistHome + "/model")

    logger.log("loading train data")
    var trainRDD = sc.parallelize(loader.trainImages.zip(loader.trainLabels))
    logger.log("repartition data")
    trainRDD = trainRDD.repartition(numWorkers)
    val numTrainData = trainRDD.count()
    logger.log("numTrainData = " + numTrainData.toString)

    // each executor (worker) will own a Solver
    val workers = sc.parallelize(Array.fill(numWorkers)(new Solver(args, mnistHome)), numWorkers).cache()
    // number of training images for each executor
    var trainPartitionSizes = trainRDD.mapPartitions(iter => Iterator.single(iter.size), true).cache()
    // training image data for each executor
    var trainPartitionMem   = trainRDD.mapPartitions(iter => Iterator.single(makeFloatPointer(iter)), true).cache()
    logger.log("trainPartitionSizes = " + trainPartitionSizes.collect().deep.toString)

    // master will owner a testing solver
    import org.bytedeco.javacpp.Loader
    import org.bytedeco.javacpp.caffe
    Loader.load(classOf[caffe])
    val testSolver = new Solver(args, mnistHome)
    val (testData,testLabl) = makeFloatPointer(loader.testImages.zip(loader.testLabels).iterator)
    val testSize = loader.testImages.length

    logger.log("start obtaining initial net weights.")
    // initialize weights (weight of the 1st executor) on master
    var netWeights = workers.map(caffeSolver => caffeSolver.instance.getWeights()).collect()(0)
    logger.log("inital net weights obtained.")

    var i = 0
    while (true) {
      logger.log("broadcasting weights", i)
      val broadcastWeights = sc.broadcast(netWeights)
      logger.log("setting weights on workers", i)
      workers.foreach(caffeSolver => caffeSolver.instance.setWeights(broadcastWeights.value))

      if (i % num_iters_per_round == 0) {
        logger.log("testing", i)
        testSolver.instance.setWeights(netWeights)
        // request the network on master to use the testing data
        testSolver.instance.setData(testData, testLabl, testSize)
        var accuracy = 0F
        // each call of ForwardPrefilled() consumes `testBatchSize` of data instances
        // so call `size / testBatchSize` many times.
        val round = testSize / testBatchSize
        for (_ <- 1 to round) {
          testSolver.instance.Forward()
          val out = testSolver.instance.getBlobs(List("accuracy"))
          accuracy += out("accuracy").data(0)
        }
        accuracy = accuracy / round
        logger.log("%.2f".format(100F * accuracy) + "% accuracy", i)
      }

      logger.log("training", i)
      workers.zipPartitions(trainPartitionSizes, trainPartitionMem) {
        case (svIt, szIt, dtIt) => 
          val caffeSolver = svIt.next
          val (data, labels) = dtIt.next
          val size = szIt.next
          val t1 = System.currentTimeMillis()
          // request the network use the testing data         
          caffeSolver.instance.setData(data, labels, size)
          // each call of Step() consumes `testBatchSize` of data instances, 
          // for we set `iter_size` in the prototxt to be 1.
          // we run only <num_batches_per_iter> per iteration.
          caffeSolver.instance.Step(num_batches_per_iter)
          val t2 = System.currentTimeMillis()
          print(s"iters took ${((t2 - t1) * 1F / 1000F).toString}s, # batches ${size / trainBatchSize}\n")
          Iterator.single(())
      }.count()
      logger.log("collecting weights", i)
      // collect all weights of all executors
      netWeights = workers.map(caffeSolver => { caffeSolver.instance.getWeights() }).reduce((a, b) => CaffeWeightCollection.add(a, b))
      // and calculate the average.
      CaffeWeightCollection.scalarDivide(netWeights, 1F * numWorkers)
      logger.log("weight = " + netWeights("conv1")(0).data(0).toString, i)

      // re-shuffle of the data does not improve the converge speed...
      // logger.log("shuffle the data...", i)
      // trainRDD = trainRDD.repartition(numWorkers)
      // trainPartitionSizes = trainRDD.mapPartitions(iter => Iterator.single(iter.size), true)
      // trainPartitionMem   = trainRDD.mapPartitions(iter => Iterator.single(makeFloatPointer(iter)), true)

      i += 1
    }

    logger.log("finished training")
  }

  // the java wrapper of MemoryDataLayer::Reset provides two alternatives, Array[Float] or 
  // java.nio.FloatBuffer, however neither works because of GC.
  // 
  // We need a piece of pinned storage.
  // 
  def makeFloatPointer(iter: Iterator[(Array[Float], Float)]): (FloatPointer, FloatPointer) = {
    val (data, labl) = iter.toArray.unzip
    val dataflatten = data.flatten
    val datanativ   = new FloatPointer(dataflatten :_*)
    val lablnativ   = new FloatPointer(labl :_*)
    (datanativ, lablnativ)
  }
}
