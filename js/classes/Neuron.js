/*

  Neuron Class

*/
class Neuron {
    constructor(activationType) {
        this.connectionTo = [];
        this.connectionFrom = [];
        this.activationType = activationType;
    }
    linkTo(neuron, weight) {
        if (weight < -1 || weight > 1) {
            throw new Error(`Neuron strength must be in the range [-1, 1]. Got: ${weight}`);
        }
        this.connectionTo.push({ neuron: neuron, weight: weight });
        neuron.connectionFrom.push({ neuron: this, weight: weight });
    }
    setValue(value) {
        this.value = ActivationTypes[this.activationType](value);
    }
    calculateValue() {
        let value = 0;
        let totalWeight = 0;
        this.connectionFrom.forEach(connection => {
            value += connection.weight * connection.neuron.value;
            totalWeight += connection.weight;
        });
        value /= totalWeight;
        this.setValue(ActivationTypes[this.activationType](value));
    }
}
/*

  ActivationTypes

*/
const ActivationTypes = {
    sigmoid: (x) => 2 / (1 + Math.pow(Math.E, (-x))) - 1,
    tanh: (x) => Math.tanh(x),
    tanh_prime: (x) => 1 / (Math.pow(Math.cosh(x), 2)),
    invtanh: (x) => Math.atanh(x),
    linear: (x) => x,
    relu: (x) => (x < 0) ? 0 : x,
    step: (x) => (x < 0) ? -1 : 1,
    stepPositive: (x) => (x < 0) ? -1 : 1
};
// Test code
let testCode = () => {
    // Inputs
    let inputTopLeft = new Neuron('linear');
    let inputTopRight = new Neuron('linear');
    let inputBottomLeft = new Neuron('linear');
    let inputBottomRight = new Neuron('linear');
    // 2nd Layer
    let neuron1 = new Neuron('sigmoid');
    inputTopLeft.linkTo(neuron1, 1);
    inputBottomLeft.linkTo(neuron1, 1);
    let neuron2 = new Neuron('sigmoid');
    inputTopRight.linkTo(neuron2, 1);
    inputBottomRight.linkTo(neuron2, 1);
    let neuron3 = new Neuron('sigmoid');
    inputTopLeft.linkTo(neuron3, 1);
    inputBottomLeft.linkTo(neuron3, -1);
    let neuron4 = new Neuron('sigmoid');
    inputTopRight.linkTo(neuron4, 1);
    inputBottomRight.linkTo(neuron4, -1);
    // 3rd Layer
    let neuron5 = new Neuron('sigmoid');
    neuron1.linkTo(neuron5, 1);
    neuron2.linkTo(neuron5, 1);
    let neuron6 = new Neuron('sigmoid');
    neuron1.linkTo(neuron6, -1);
    neuron2.linkTo(neuron6, 1);
    let neuron7 = new Neuron('sigmoid');
    neuron3.linkTo(neuron7, 1);
    neuron4.linkTo(neuron7, -1);
    let neuron8 = new Neuron('sigmoid');
    neuron3.linkTo(neuron8, 1);
    neuron4.linkTo(neuron8, 1);
    // Set input:
    inputTopLeft.setValue(1);
    inputTopRight.setValue(-1);
    inputBottomLeft.setValue(-1);
    inputBottomRight.setValue(1);
    // Calc layer 2
    neuron1.calculateValue();
    neuron2.calculateValue();
    neuron3.calculateValue();
    neuron4.calculateValue();
    // Calc layer 3
    neuron5.calculateValue();
    neuron6.calculateValue();
    neuron7.calculateValue();
    neuron8.calculateValue();
};
