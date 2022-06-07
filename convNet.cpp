//
//  convNet_code.cpp
//  convNet_code
//
//  Created by Kevin Du on 1/17/22.
//


#include "snake.h"

double squ(double x){
    return x * x;
}

int max(int x, int y){
    if(x < y) return y;
    return x;
}

double randWeight(double startingParameterRange){
    return (((double)rand() / RAND_MAX)*2-1) * startingParameterRange;
}

double nonlinear(double x){
    if(x>0) return x;
    return x*0.1;
}

double dinvnonlinear(double x){
    if(x>0) return 1;
    return 0.1;
}

// Layer

void Layer::setupParams(){
    numParams = numWeights + numBias;
    //params = new double[numParams];
    params = (double*) malloc(numParams * sizeof(double));
    assert(params != NULL);
    weights = params;
    bias = params + numWeights;
    //Dparams = new double[numParams];
    Dparams = (double*) malloc(numParams * sizeof(double));
    assert(Dparams != NULL);
    Dweights = Dparams;
    Dbias = Dparams + numWeights;
}

void Layer::randomize(double startingParameterRange){
    for(int i=0; i<numParams; i++){
        params[i] = randWeight(startingParameterRange);
    }
}

void Layer::resetGradient(){
    for(int i=0; i<numParams; i++){
        Dparams[i] = 0;
    }
}

void Layer::updateParameters(double mult, double momentum){
    for(int i=0; i<numParams; i++){
        params[i] -= Dparams[i] * mult;
        Dparams[i] *= momentum;
    }
    // Regularize
    double sum = 0;
    for(int i=0; i<numParams; i++){
        sum += squ(params[i]);
    }
    //cout<<"Param sum: "<<sum<<'\n';
    /*
    if(sum <= maxNorm) return;
    for(int i=0; i<numParams; i++){
        params[i] *= sqrt(maxNorm / sum);
    }*/
}

void Layer::save(){
    for(int i=0; i<numParams; i++){
        (*netOut)<<params[i]<<' ';
    }
    (*netOut)<<"\n\n";
}

void Layer::readNet(){
    for(int i=0; i<numParams; i++){
        (*netIn)>>params[i];
    }
}


// ConvLayer

ConvLayer::ConvLayer(int inD, int inH, int inW, int outD, int outH, int outW, int convH, int convW){
    inputDepth = inD;
    inputHeight = inH;
    inputWidth = inW;
    outputDepth = outD;
    outputHeight = outH;
    outputWidth = outW;
    convHeight = convH;
    convWidth = convW;
    
    shiftr = (inputHeight - outputHeight - convHeight + 1) / 2;
    shiftc = (inputWidth - outputWidth - convWidth + 1) / 2;
    w1 = outputDepth * convHeight * convWidth;
    w2 = convHeight * convWidth;
    w3 = convWidth;
    
    numWeights = inputDepth * outputDepth * convHeight * convWidth;
    numBias = outputDepth;
    this->setupParams();
}

void ConvLayer::pass(double* inputs, double* outputs){
    double sum;
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                sum = bias[j];
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < inputHeight && inputc >= 0 && inputc < inputWidth){
                            for(int i=0; i<inputDepth; i++){
                                sum += inputs[i*inputHeight*inputWidth + inputr*inputWidth + inputc] * weights[i*w1 + j*w2 + r*w3 + c];
                            }
                        }
                    }
                }
                outputs[j*outputHeight*outputWidth + x*outputWidth + y] = nonlinear(sum);
            }
        }
    }
}

void ConvLayer::backProp(double* inputs, double* Dinputs, double* Doutputs){
    for(int i=0; i<inputDepth*inputHeight*inputWidth; i++){
        Dinputs[i] = 0;
    }
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < inputHeight && inputc >= 0 && inputc < inputWidth){
                            for(int i=0; i<inputDepth; i++){
                                Dinputs[i*inputHeight*inputWidth + inputr*inputWidth + inputc] += weights[i*w1 + j*w2 + r*w3 + c] * Doutputs[j*outputHeight*outputWidth + x*outputWidth + y];
                            }
                        }
                    }
                }
            }
        }
    }
    for(int i=0; i<inputDepth*inputHeight*inputWidth; i++){
        Dinputs[i] *= dinvnonlinear(inputs[i]);
    }
}

void ConvLayer::accumulateGradient(double* inputs, double* Doutputs){
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                double Dout = Doutputs[j*outputHeight*outputWidth + x*outputWidth + y];
                Dbias[j] += Dout;
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < inputHeight && inputc >= 0 && inputc < inputWidth){
                            for(int i=0; i<inputDepth; i++){
                                Dweights[i*w1 + j*w2 + r*w3 + c] += inputs[i*inputHeight*inputWidth + inputr*inputWidth + inputc] * Dout;
                            }
                        }
                    }
                }
            }
        }
    }
}


// PoolLayer

PoolLayer::PoolLayer(int inD, int inH, int inW, int outD, int outH, int outW){
    inputDepth = inD;
    inputHeight = inH;
    inputWidth = inW;
    outputDepth = outD;
    outputHeight = outH;
    outputWidth = outW;
    
    numWeights = 0;
    numBias = 0;
    numParams = 0;
    
    maxIndices = new int[outD * outH * outW];
}

void PoolLayer::pass(double* inputs, double* outputs){
    double maxVal,candVal;
    int maxIndex;
    int index;
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                maxVal = -100000;
                maxIndex = -1;
                for(int r=0; r<2; r++){
                    for(int c=0; c<2; c++){
                        index = j*inputHeight*inputWidth + (2*x+r)*inputWidth + (2*y+c);
                        candVal = inputs[index];
                        if(maxVal < candVal){
                            maxVal = candVal;
                            maxIndex = index;
                        }
                    }
                }
                index = j*outputHeight*outputWidth + x*outputWidth + y;
                outputs[index] = maxVal;
                maxIndices[index] = maxIndex;
            }
        }
    }
}

void PoolLayer::backProp(double* inputs, double* Dinputs, double* Doutputs){
    for(int i=0; i<inputDepth*inputHeight*inputWidth; i++){
        Dinputs[i] = 0;
    }
    for(int i=0; i<outputDepth*outputHeight*outputWidth; i++){
        Dinputs[maxIndices[i]] = Doutputs[i];
    }
}


// DenseLayer

DenseLayer::DenseLayer(int inSize, int outSize){
    inputSize = inSize;
    outputSize = outSize;
    
    numWeights = inputSize * outputSize;
    numBias = outputSize;
    this->setupParams();
}

void DenseLayer::pass(double* inputs, double* outputs){
    double sum;
    for(int i=0; i<outputSize; i++){
        sum = bias[i];
        for(int j=0; j<inputSize; j++){
            sum += weights[j*outputSize + i] * inputs[j];
        }
        outputs[i] = nonlinear(sum);
    }
}

void DenseLayer::backProp(double* inputs, double* Dinputs, double* Doutputs){
    double sum;
    for(int i=0; i<inputSize; i++){
        sum = 0;
        for(int j=0; j<outputSize; j++){
            sum += weights[i*outputSize + j] * Doutputs[j];
        }
        Dinputs[i] = sum * dinvnonlinear(inputs[i]);
    }
}

void DenseLayer::accumulateGradient(double* inputs, double* Doutputs){
    for(int i=0; i<outputSize; i++){
        Dbias[i] += Doutputs[i];
        for(int j=0; j<inputSize; j++){
            Dweights[j*outputSize + i] += Doutputs[i] * inputs[j];
        }
    }
}


// ConvNet

void Agent::initInput(int depth, int height, int width, int convHeight, int convWidth){
    input = new networkInput;
    layerHold.push_back(new InputLayer(depth, height, width, convHeight, convWidth, input));
    prevDepth = depth;
    prevHeight = height;
    prevWidth = width;
    maxNodes = max(maxNodes, depth * height * width);
}

void Agent::addConvLayer(int depth, int height, int width, int convHeight, int convWidth){
    layerHold.push_back(new ConvLayer(prevDepth, prevHeight, prevWidth, depth, height, width, convHeight, convWidth));
    prevDepth = depth;
    prevHeight = height;
    prevWidth = width;
    maxNodes = max(maxNodes, depth * height * width);
}

void Agent::addPoolLayer(int depth, int height, int width){
    layerHold.push_back(new PoolLayer(prevDepth, prevHeight, prevWidth, depth, height, width));
    prevDepth = depth;
    prevHeight = height;
    prevWidth = width;
    maxNodes = max(maxNodes, depth * height * width);
}

void Agent::addDenseLayer(int numNodes){
    layerHold.push_back(new DenseLayer(prevDepth * prevHeight * prevWidth, numNodes));
    prevDepth = numNodes;
    prevHeight = 1;
    prevWidth = 1;
    maxNodes = max(maxNodes, numNodes);
}

void Agent::randomize(double startingParameterRange){
    for(int l=0; l<numLayers; l++){
        layers[l]->randomize(startingParameterRange);
    }
}

void Agent::pass(){
    layers[0]->pass(NULL, activation[1]);
    for(int l=1; l<numLayers; l++){
        layers[l]->pass(activation[l], activation[l+1]);
    }
    output = activation[numLayers][0];
}

void Agent::resetGradient(){
    for(int l=0; l<numLayers; l++){
        layers[l]->resetGradient();
    }
}

void Agent::backProp(){
    pass();
    Dbias[numLayers-1][0] = 2 * (activation[numLayers][0] - expected) * dinvnonlinear(activation[numLayers][0]);
    for(int l=numLayers-1; l>0; l--){
        layers[l]->accumulateGradient(activation[l], Dbias[l]);
        layers[l]->backProp(activation[l], Dbias[l-1], Dbias[l]);
    }
    layers[0]->accumulateGradient(NULL, Dbias[0]);
}

void Agent::updateParameters(double mult, double momentum){
    for(int l=0; l<numLayers; l++){
        layers[l]->updateParameters(mult, momentum);
    }
}

void Agent::save(string fileName){
    netOut = ofstream(fileName);
    for(int l=0; l<numLayers; l++){
        layers[l]->save();
    }
    netOut.close();
}

void Agent::readNet(string fileName){
    netIn = ifstream(fileName);
    for(int l=0; l<numLayers; l++){
        layers[l]->readNet();
    }
    netIn.close();
}

void Agent::quickSetup(){
    numLayers = layerHold.size();
    layers = new Layer*[numLayers];
    for(int i=0; i<numLayers; i++){
        layers[i] = layerHold[i];
    }
    activation = new double*[numLayers + 1];
    Dbias = new double*[numLayers];
    for(int l=0; l<=numLayers; l++){
        activation[l] = new double[maxNodes];
    }
    for(int l=0; l<numLayers; l++){
        Dbias[l] = new double[maxNodes];
    }
    for(int l=0; l<numLayers; l++){
        layers[l]->netIn = &netIn;
        layers[l]->netOut = &netOut;
    }
    resetGradient();
}
