
class CaffeWeightCollection(private val col: Array[(String, Array[FloatNDArray])]) extends Serializable {
  def layer(name: String): Array[FloatNDArray] = {
    col.find(_._1 == name).get._2
  }
  def numLayers = col.length

  def info() {
    println(s"Num of layers: ${col.length}")
    for ((_, layer) <- col) {
      println(s"layer num of NDArray: ${layer.length}")
      for (arr <- layer) {
        println(s"  shape: ${arr.shape.deep}")
      }
    }
    val arrs = col.map(_._2).flatten
    var count = 0
    for (arr <- arrs) {
      count += arr.shape.product
    }
    println(s"total size: ${count}")
  }
}

object CaffeWeightCollection {
  
  def apply(): CaffeWeightCollection = {
    new CaffeWeightCollection(Array())
  }

  def apply(iter: Iterable[(String, Array[FloatNDArray])]): CaffeWeightCollection = {
    new CaffeWeightCollection(iter.toArray)
  }

  def scalarDivide(weights: CaffeWeightCollection, v: Float): Unit = {
    for (layer <- weights.col) {
      for (param <- layer._2) {
        param.scalarDivide(v)
      }
    }
  }

  def add(weights1: CaffeWeightCollection, weights2: CaffeWeightCollection): CaffeWeightCollection = {
    assert(weights1.numLayers == weights2.numLayers)
    val newWeights = 
      weights1.col.indices map {
        case ind => 
          val (name1, layer1) = weights1.col(ind)
          val (name2, layer2) = weights2.col(ind)
          assert(name1 == name2)
          val layer = layer1 zip layer2 map {
            case (a,b) => {
              assert(a.shape.deep == b.shape.deep)
              FloatNDArray.plus(a,b)
            }
          }
          (name1, layer)
      }
    CaffeWeightCollection(newWeights)
  }
}
