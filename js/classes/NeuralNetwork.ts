/*

NeuralNetwork Class

*/

type InputLayer = {
  neurons?: number
}

type ConnectedLayer = {
  neurons?: number
  weights: Matrix,
  bias: number
}

class NeuralNetwork {
  input: Vector
  inputLayer: InputLayer
  hiddenLayers: ConnectedLayer[]
  outputLayer: ConnectedLayer
  activations: Vector[]
  learningRate: number
  activationFunction: (x: number) => number
  activationFunctionInv: (x: number) => number
  activationFunctionPrime: (x: number) => number

  constructor(inputLayer: InputLayer, hiddenLayers: ConnectedLayer[], outputLayer: ConnectedLayer) {
    this.inputLayer = inputLayer
    this.hiddenLayers = hiddenLayers
    this.outputLayer = outputLayer

    this.learningRate = 0.5

    this.activationFunction = x => Math.tanh(x)
    this.activationFunctionInv = x => Math.atanh(x)
    this.activationFunctionPrime = x => 1 / (Math.cosh(x) ** 2)
  }

  setInput(input: Vector) {
    this.input = input
  }

  run() {
    let currentLayerVector = this.input
    this.activations = []
    this.activations[0] = currentLayerVector

    for (let i = 0; i < this.hiddenLayers.length; i++) {
      currentLayerVector = currentLayerVector
      .multiplyWithMatrix(this.hiddenLayers[i].weights)
      .each(x => x + this.hiddenLayers[i].bias)
      .each(this.activationFunction)

      this.activations[i + 1] = currentLayerVector
    }

    currentLayerVector = currentLayerVector
    .multiplyWithMatrix(this.outputLayer.weights)
    .each(x => x + this.outputLayer.bias)
    .each(this.activationFunction)

    this.activations[this.hiddenLayers.length + 1] = currentLayerVector

    return currentLayerVector
  }

  calculateCost(output: Vector, expectedOutput: Vector) {
    let cost = output
    .subtract(expectedOutput)
    .each(x => x ** 2).sum
    return cost
  }

  train(input: Vector, expectedOutput: Vector) {
    // Forward propagation
    this.setInput(input)
    let output = this.run()

    // Calculate cost
    let cost = this.calculateCost(output, expectedOutput)
    console.log('cost: ' + cost)

    // Backward propagation

    // BP based on weight change

    for (let j = 0; j < this.outputLayer.weights.rows; j++) {
      for (let k = 0; k < this.outputLayer.weights.cols; k++) {
        let change = this.learningRate * this.activations[this.layers - 2].values[k] * this.activationFunctionPrime(this.activationFunctionInv(this.activations[this.layers - 1].values[j])) * 2 * (this.activations[this.layers - 1].values[j] - expectedOutput.values[j])

        this.outputLayer.weights.setElement(j, k, this.outputLayer.weights.getElement(j, k) - change)
        console.log('weight change:' + change)
      }
    }

    // BP based on activation of previous layer

    let outputLayerChangeArr: number[] = []

    for (let j = 0; j < this.outputLayer.weights.rows; j++) {
      outputLayerChangeArr[j] = 0

      for (let k = 0; k < this.outputLayer.weights.cols; k++) {
        let change = this.learningRate * this.outputLayer.weights.getElement(j, k) * this.activationFunctionPrime(this.activationFunctionInv(this.activations[this.layers - 1].values[j])) * 2 * (this.activations[this.layers - 1].values[j] - expectedOutput.values[j])

        outputLayerChangeArr[j] += change
        console.log('activation change: ' + change)
      }
    }

    let expectedActivations: (Vector)[] = new Array(this.layers)
    expectedActivations[this.layers - 1] = expectedOutput

    for (let i = this.hiddenLayers.length - 1; i >= 0; i--) {
      let hiddenLayersChangeArr: number[][] = []

      for (let j = 0; j < this.hiddenLayers[i].weights.rows; j++) {
        hiddenLayersChangeArr[i][j] = 0

        for (let k = 0; k < this.hiddenLayers[i].weights.cols; k++) {
          let change = this.learningRate * this.hiddenLayers[i].weights.getElement(j, k) * this.activationFunctionPrime(this.activationFunctionInv(this.activations[this.layers - 1 - i].values[j])) * 2 * (this.activations[this.layers - 1 - i].values[j] - expectedActivations[this.layers - 1 - i].values[j])

          hiddenLayersChangeArr[i][j] += change
          console.log('activation change: ' + change)
        }
      }
    }

    // BP based on bias

    // Calculate cost again
    let output_after = this.run()
    let cost_after = this.calculateCost(output_after, expectedOutput)
    console.log('cost: ' + cost_after)
  }

  static createRandom(layers: number[]) {
    let hiddenLayers: ConnectedLayer[] = []

    for (let i = 1; i < layers.length - 1; i++) {
      hiddenLayers.push({
        weights: Matrix.createRandom(layers[i], layers[i - 1]),
        bias: Math.random() * 2 - 1
      })
    }

    let outputLayer = {
      weights: Matrix.createRandom(layers[layers.length - 1], layers[layers.length - 2]),
      bias: Math.random() * 2 - 1
    }

    return new NeuralNetwork({}, hiddenLayers, outputLayer)
  }

  get layers() {
    return this.hiddenLayers.length + 2
  }
}

const sigmoid = (n: number) => {
  return 2 / (1 + Math.E ** (-n)) - 1
}

// Test code
() => {
  new NeuralNetwork(
    {
      neurons: 4
    },
    [
      {
        neurons: 4,
        weights: new Matrix([
          [ 1, 0, 1, 0 ],
          [ 0, 1, 0, 1 ],
          [ 1, 0, -1, 0 ],
          [ 0, 1, 0, -1 ]
        ]),
        bias: 0
      },
      {
        neurons: 4,
        weights: new Matrix([
          [ 1, 1, 0, 0 ],
          [ -1, 1, 0, 0 ],
          [ 0, 0, 1, -1 ],
          [ 0, 0, 1, 1 ]
        ]),
        bias: 0
      }
    ],
    {
      neurons: 8,
      weights: new Matrix([
        [ 1, 0, 0, 0 ],
        [ -1, 0, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, -1, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, -1, 0 ],
        [ 0, 0, 0, 1 ],
        [ 0, 0, 0, -1 ]
      ]),
      bias: 0
    }
  )
}
