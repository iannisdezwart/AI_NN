/*

  Vector Class

*/
class Vector {
    constructor(...values) {
        this.values = values;
    }
    get dimension() {
        return this.values.length;
    }
    add(vector2) {
        if (this.dimension != vector2.dimension) {
            throw new Error(`Can't add vectors with different dimensions.`);
        }
        let newValues = [];
        for (let i = 0; i < this.values.length; i++) {
            newValues[i] = this.values[i] + vector2.values[i];
        }
        return new Vector(...newValues);
    }
    subtract(vector2) {
        if (this.dimension != vector2.dimension) {
            throw new Error(`Can't subtract vectors with different dimensions.`);
        }
        let newValues = [];
        for (let i = 0; i < this.values.length; i++) {
            newValues[i] = this.values[i] - vector2.values[i];
        }
        return new Vector(...newValues);
    }
    multiply(multiplier) {
        if (multiplier.constructor.name == 'Number') {
            return this.multiplyWithScalar(multiplier);
        }
        if (multiplier.constructor.name == 'Vector') {
            return this.dotProduct(multiplier);
        }
        if (multiplier.constructor.name == 'Matrix') {
            return this.multiplyWithMatrix(multiplier);
        }
    }
    divide(scalar) {
        let newValues = [];
        for (let i = 0; i < this.values.length; i++) {
            newValues[i] = this.values[i] / scalar;
        }
        return new Vector(...newValues);
    }
    multiplyWithScalar(scalar) {
        let newValues = [];
        for (let i = 0; i < this.values.length; i++) {
            newValues[i] = this.values[i] * scalar;
        }
        return new Vector(...newValues);
    }
    dotProduct(vector2) {
        if (this.dimension != vector2.dimension) {
            throw new Error(`Can't calculate the dot product of Vectors with different dimensions.`);
        }
        let sum = 0;
        for (let i = 0; i < this.values.length; i++) {
            sum += this.values[i] * vector2.values[i];
        }
        return sum;
    }
    multiplyWithMatrix(matrix) {
        return matrix.multiplyWithVector(this);
    }
    each(f) {
        let newValues = [];
        for (let i = 0; i < this.values.length; i++) {
            newValues[i] = f(this.values[i]);
        }
        return new Vector(...newValues);
    }
    distance(vector2) {
        if (this.dimension != vector2.dimension) {
            throw new Error(`Can't calculate the distance between Vectors with different dimensions.`);
        }
        let sum = 0;
        for (let i = 0; i < this.values.length; i++) {
            sum += Math.pow((this.values[i] - vector2.values[i]), 2);
        }
        return Math.sqrt(sum);
    }
    get sum() {
        let sum = 0;
        for (let i = 0; i < this.values.length; i++) {
            sum += this.values[i];
        }
        return sum;
    }
    get angle() {
        if (this.dimension == 2) {
            return Math.atan(this.values[1] / this.values[0]);
        }
        else {
            return new Error(`Can't calculate angle of a ${this.dimension}-dimensional Vector.`);
        }
    }
    get magnitude() {
        return Math.hypot(...this.values);
    }
    static fromAngle(angle) {
        return new Vector(Math.cos(angle), Math.sin(angle));
    }
    static createRandom(dimension) {
        let values = new Array(dimension);
        for (let i = 0; i < dimension; i++) {
            values[i] = Math.random() * 2 - 1;
        }
        return new Vector(...values);
    }
}
