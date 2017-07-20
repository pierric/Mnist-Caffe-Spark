import scala.collection.mutable.Map
import scala.collection.mutable.MutableList

object CaffeWeightCollection {
  def scalarDivide(weights: Map[String, MutableList[FloatNDArray]], v: Float): Unit = {
    for (name <- weights.keys) {
      for (j <- 0 to weights(name).length - 1) {
        weights(name)(j).scalarDivide(v)
      }
    }
  }

  def add(weights1: Map[String, MutableList[FloatNDArray]], weights2: Map[String, MutableList[FloatNDArray]]): Map[String, MutableList[FloatNDArray]] = {
    if (weights1.keys != weights2.keys) {
      throw new Exception("weights1.keys != weights2.keys, weights1.keys = " + weights1.keys.toString + ", and weights2.keys = " + weights2.keys.toString + "\n")
    }
    val newWeights = Map[String, MutableList[FloatNDArray]]()
    for (name <- weights1.keys) {
      newWeights += (name -> MutableList())
      if (weights1(name).length != weights2(name).length) {
        throw new Exception("weights1(name).length != weights2(name).length, name = " + name + ", weights1(name).length = " + weights1(name).length.toString + ", weights2(name).length = " + weights2(name).length.toString)
      }
      for (j <- 0 to weights1(name).length - 1) {
        if (weights1(name)(j).shape.deep != weights2(name)(j).shape.deep) {
          throw new Exception("weights1(name)(j).shape != weights2(name)(j).shape, name = " + name + ", j = " + j.toString + ", weights1(name)(j).shape = " + weights1(name)(j).shape.deep.toString + ", weights2(name)(j).shape = " + weights2(name)(j).shape.deep.toString)
        }
        newWeights(name) += FloatNDArray.plus(weights1(name)(j), weights2(name)(j))
      }
    }
    newWeights
  }
}
