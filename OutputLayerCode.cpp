//
//  OutputLayerCode.cpp
//  OutputLayerCode
//
//  Created by Kevin Du on 6/1/22.
//

#include "snake.h"

OutputLayer::OutputLayer(int inSize, int outSize, double* _output, double* _expected){
    inputSize = inSize;
    outputSize = outSize;
    output = _output;
    expected = _expected;
    
    numWeights = inputSize * outputSize;
    numBias = outputSize;
    this->setupParams();
}

void OutputLayer::pass(double* inputs, double* outputs){
    double sum;
    for(int i=0; i<outputSize; i++){
        sum = bias[i];
        for(int j=0; j<inputSize; j++){
            sum += weights[j*outputSize + i] * inputs[j];
        }
        if(i == 0){
            outputs[i] = nonlinear(sum);
        }
        else{
            outputs[i] = unit(sum);
        }
    }
}

void OutputLayer::accumulateGradient(double* inputs, double* Doutputs){
    for(int i=0; i<outputSize; i++){
        double factor;
        if(i == 0){
            factor = dinvnonlinear(output[i]);
        }
        else{
            if(output[i] < 0){
                factor = 0;
            }
            else{
                factor = dinvunit(output[i]) * 5;
            }
        }
        Doutputs[i] = 2 * (output[i] - expected[i]) * factor;
    }
    for(int i=0; i<outputSize; i++){
        Dbias[i] += Doutputs[i];
        for(int j=0; j<inputSize; j++){
            Dweights[j*outputSize + i] += Doutputs[i] * inputs[j];
        }
    }
}

void OutputLayer::backProp(double* inputs, double* Dinputs, double* Doutputs){
    double sum;
    for(int i=0; i<inputSize; i++){
        sum = 0;
        for(int j=0; j<outputSize; j++){
            sum += weights[i*outputSize + j] * Doutputs[j];
        }
        Dinputs[i] = sum * dinvnonlinear(inputs[i]);
    }
}
