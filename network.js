function activation(x) {
  //return Math.max(0, x);
  return x / (1 + Math.exp(-x));
}

class Layer {
  constructor(inputCount, outputCount) {
    this.inputCount = inputCount;
    this.outputCount = outputCount;
    this.weights = new Array(inputCount);
    this.biases = new Array(outputCount);

    for (let i = 0; i < outputCount; i++) {
      this.biases[i] = 0;
    }

    for (let i = 0; i < inputCount; i++) {
      this.weights[i] = new Array(outputCount);
      for (let j = 0; j < outputCount; j++) {
        this.weights[i][j] = 0;
      }
    }
  }

  calculate(inputValues) {
    if (inputValues.length !== this.inputCount) {
      console.log("Input length mismatch");
      return;
    }

    let outputs = new Array(this.outputCount);

    for (let i = 0; i < this.outputCount; i++) {
      let sum = 0;
      for (let j = 0; j < this.inputCount; j++) {
        sum += this.weights[j][i] * inputValues[j];
      }
      sum += this.biases[i];
      outputs[i] = activation(sum);
    }

    return outputs;
  }
}

class NeuralNetwork {
  constructor(layerCounts) {
    this.layerCounts = layerCounts;
    this.layers = new Array(this.layerCounts.length - 1);
    this.learningRate = 0.1;
    for (let i = 0; i < this.layerCounts.length - 1; i++) {
      this.layers[i] = new Layer(layerCounts[i], layerCounts[i + 1]);
    }
  }

  train(trainingData, learningRate) {
    this.learningRate = learningRate;
    let loss = 0;
    for (let i = 0; i < trainingData.length; i++) {
      let current = trainingData[i];
      let currentloss = current[1] - this.run([current[0]]);
      loss += currentloss * currentloss;
    }
    console.log("loss: " + loss);
    for (let layer of this.layers) {
      for (let weights of layer.weights) {
        for (let i = 0; i < weights.length; i++) {
          //console.log(weights[i]);
          //this.learningRate  = Math.random()*2-1
          let temp = Math.random() * this.learningRate - this.learningRate / 2;
          weights[i] += temp;
          let temploss = 0;
          for (let i = 0; i < trainingData.length; i++) {
            let current = trainingData[i];
            let currentloss = current[1] - this.run([current[0]]);
            temploss += currentloss * currentloss;
          }
          if (temploss > loss) {
            weights[i] -= temp;
          }
        }
      }

      for (let i = 0; i < layer.biases.length; i++) {
        //console.log(layer.biases[i]);
         //this.learningRate  = Math.random()*2-1
        let temp = Math.random() * this.learningRate - this.learningRate / 2;
        layer.biases[i] += temp;
        let temploss = 0;
        for (let i = 0; i < trainingData.length; i++) {
          let current = trainingData[i];
          let currentloss = current[1] - this.run([current[0]]);
          temploss += currentloss * currentloss;
        }
        if (temploss > loss) {
          layer.biases[i] -= temp;
        }
      }
    }
  }

  run(inputValues) {
    let tempInput = [...inputValues];

    for (let i = 0; i < this.layerCounts.length - 1; i++) {
      tempInput = this.layers[i].calculate(tempInput);
    }

    return tempInput;
  }
}
