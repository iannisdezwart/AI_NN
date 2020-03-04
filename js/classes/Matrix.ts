/*

  Matrix Class

*/

class Matrix {
  values: number[][]
  rows: number
  cols: number

  constructor(matrix: number[][]) {
    this.rows = matrix.length
    this.cols = matrix[0].length

    for (let i = 1; i < matrix.length; i++) {
      if (matrix[i].length != this.cols) {
        throw new Error(`A Matrix must have the same number of elements in each row.`)
      }
    }

    this.values = matrix
  }

  getElement(row: number, col: number) {
    return this.values[row][col]
  }

  setElement(row: number, col: number, value: number) {
    this.values[row][col] = value
  }

  getRow(row: number) {
    return this.values[row]
  }

  getCol(col: number) {
    let values = []

    for (let i = 0; i < this.values.length; i++) {
      values[i] = this.values[i][col]
    }

    return values
  }

  setRow(row: number, arr: number[]) {
    this.values[row] = arr
  }

  setCol(col: number, arr: number[]) {
    for (let i = 0; i < this.values.length; i++) {
      this.values[i][col] = arr[i]
    }
  }

  copy() {
    return new Matrix(this.values)
  }

  matches(matrix2: Matrix) {
    for (let i = 0; i < this.values.length; i++) {
      for (let j = 0; j < this.values[i].length; j++) {
        if (this.values[i][j] != matrix2.values[i][j]) {
          return false
        }
      }
    }

    return true
  }

  multiply(multiplier: any) {
    if (multiplier.constructor.name == 'Number') {
      return this.multiplyWithScalar(multiplier)
    }
    if (multiplier.constructor.name == 'Vector') {
      return this.multiplyWithVector(multiplier)
    }
  }

  multiplyWithScalar(scalar: number) {
    let outputMatrix = this.copy()
    for (let row = 0; row < outputMatrix.values.length; row++) {
      for (let col = 0; col < outputMatrix.values[row].length; col++) {
        outputMatrix.setElement(row, col, outputMatrix.getElement(row, col) * scalar)
      }
    }
  }

  multiplyWithVector(vector: Vector) {
    if (this.cols != vector.dimension) {
      throw new Error(`Matrix cols don't match Vector dimension`)
    }

    let outVectorValues = []
    for (let row = 0; row < this.rows; row++) {
      outVectorValues[row] = 0
      for (let col = 0; col < this.cols; col++) {
        outVectorValues[row] += this.getElement(row, col) * vector.values[col]
      }
    }

    return new Vector(...outVectorValues)
  }

  each(f: (n: number) => number) {
    let outputMatrix = new Matrix(this.values)

    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.cols; col++) {
        outputMatrix.setElement(row, col, f(outputMatrix.getElement(row, col)))
      }
    }

    return outputMatrix
  }

  static createRandom(rows: number, cols: number) {
    let randomValues = []

    for (let row = 0; row < rows; row++) {
      randomValues[row] = []
      for (let col = 0; col < cols; col++) {
        randomValues[row][col] = Math.random() * 2 - 1
      }
    }

    return new Matrix(randomValues)
  }
}
