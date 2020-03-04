/*

NeuralNetwork Class

*/
class NeuralNetwork {
    constructor(inputLayer, hiddenLayers, outputLayer) {
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
    }
    setInput(input) {
        this.input = input;
    }
    run() {
        let currentLayerVector = this.input;
        this.activations = [];
        this.activations[0] = currentLayerVector;
        for (let i = 0; i < this.hiddenLayers.length; i++) {
            currentLayerVector = currentLayerVector
                .multiplyWithMatrix(this.hiddenLayers[i].weights)
                .each(x => x + this.hiddenLayers[i].bias)
                .each(ActivationTypes.tanh);
            this.activations[i + 1] = currentLayerVector;
        }
        currentLayerVector = currentLayerVector
            .multiplyWithMatrix(this.outputLayer.weights)
            .each(x => x + this.outputLayer.bias)
            .each(ActivationTypes.tanh);
        this.activations[this.hiddenLayers.length + 1] = currentLayerVector;
        return currentLayerVector;
    }
    calculateCost(output, expectedOutput) {
        let cost = output
            .subtract(expectedOutput)
            .each(x => Math.pow(x, 2)).sum;
        return cost;
    }
    train(input, expectedOutput) {
        // Forward propagation
        this.setInput(input);
        let output = this.run();
        // Calculate cost
        let cost = this.calculateCost(output, expectedOutput);
        console.log('cost: ' + cost);
        // Backward propagation
        // BP based on weight change
        for (let j = 0; j < this.outputLayer.weights.rows; j++) {
            for (let k = 0; k < this.outputLayer.weights.cols; k++) {
                let change = this.activations[this.layers - 2].values[k] * ActivationTypes.tanh_prime(ActivationTypes.invtanh(this.activations[this.layers - 1].values[j])) * 2 * (this.activations[this.layers - 1].values[j] - expectedOutput.values[j]);
                this.outputLayer.weights.setElement(j, k, this.outputLayer.weights.getElement(j, k) - change);
                console.log('weight change:' + change);
            }
        }
        // BP based on activation of previous layer
        // let expectedActivations: (Vector)[] = new Array(this.layers)
        // expectedActivations[this.layers - 1] = expectedOutput
        // for (let i = this.hiddenLayers.length - 1; i >= 0; i--) {
        //   for (let j = 0; j < this.hiddenLayers[i].weights.rows; j++) {
        //     for (let k = 0; k < this.hiddenLayers[i].weights.cols; k++) {
        //       let change = this.hiddenLayers[i].weights.getElement(j, k) * ActivationTypes.tanh_prime(ActivationTypes.invtanh(this.activations[this.layers - 1 - i].values[j])) * 2 * (this.activations[this.layers - 1 - i].values[j] - expectedActivations[this.layers - 1 - i].values[j])
        //     }
        //   }
        // }
        // let outputLayerChangeArr: number[] = []
        //
        // for (let j = 0; j < this.outputLayer.weights.rows; j++) {
        //   outputLayerChangeArr[j] = 0
        //   for (let k = 0; k < this.outputLayer.weights.cols; k++) {
        //     let change = this.outputLayer.weights.getElement(j, k) * ActivationTypes.tanh_prime(ActivationTypes.invtanh(this.activations[this.layers - 1].values[j])) * 2 * (this.activations[this.layers - 1].values[j] - expectedOutput.values[j])
        //
        //     outputLayerChangeArr[j] += change
        //     console.log('activation change: ' + change)
        //   }
        // }
        // console.log(outputLayerChangeArr)
        // let desires = new Vector(...new Array(this.activations[this.activations.length - 2].values.length).fill(0))
        //
        // for (let j = 0; j < this.activations[this.activations.length - 1].values.length; j++) {
        //   let neuron_change_wanted = this.activations[this.activations.length - 1].values[j] - expectedOutput.values[j]
        //   let desire_arr = []
        //   for (let k = 0; k < this.activations[this.activations.length - 2].values.length; k++) {
        //
        //   }
        //   desires.add(new Vector(...desire_arr))
        // }
        // Calculate cost again
        let output_after = this.run();
        let cost_after = this.calculateCost(output_after, expectedOutput);
        console.log('cost: ' + cost_after);
    }
    static createRandom(layers) {
        let hiddenLayers = [];
        for (let i = 1; i < layers.length - 1; i++) {
            hiddenLayers.push({
                weights: Matrix.createRandom(layers[i], layers[i - 1]),
                bias: Math.random() * 2 - 1
            });
        }
        let outputLayer = {
            weights: Matrix.createRandom(layers[layers.length - 1], layers[layers.length - 2]),
            bias: Math.random() * 2 - 1
        };
        return new NeuralNetwork({}, hiddenLayers, outputLayer);
    }
    get layers() {
        return this.hiddenLayers.length + 2;
    }
}
const sigmoid = (n) => {
    return 2 / (1 + Math.pow(Math.E, (-n))) - 1;
};
// Test code
() => {
    new NeuralNetwork({
        neurons: 4
    }, [
        {
            neurons: 4,
            weights: new Matrix([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, -1, 0],
                [0, 1, 0, -1]
            ]),
            bias: 0
        },
        {
            neurons: 4,
            weights: new Matrix([
                [1, 1, 0, 0],
                [-1, 1, 0, 0],
                [0, 0, 1, -1],
                [0, 0, 1, 1]
            ]),
            bias: 0
        }
    ], {
        neurons: 8,
        weights: new Matrix([
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1]
        ]),
        bias: 0
    });
};
