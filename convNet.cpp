//
//  convNet_code.cpp
//  convNet_code
//
//  Created by Kevin Du on 1/17/22.
//


#include "snake.h"

double squ(double x){
    return x*x;
}

double randWeight(){
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

// ConvLayer

void ConvLayer::initialize(){
    shiftr = (inputHeight - outputHeight - convHeight + 1) / 2;
    shiftc = (inputWidth - outputWidth - convWidth + 1) / 2;
    int i,j,r,c;
    for(i=0; i<inputDepth; i++){
        for(j=0; j<outputDepth; j++){
            for(r=0; r<convHeight; r++){
                for(c=0; c<convWidth; c++){
                    weights[i][j][r][c] = randWeight();
                }
            }
        }
    }
    for(i=0; i<outputDepth; i++){
        bias[i] = randWeight();
    }
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
                                sum += inputs[i*inputHeight*inputWidth + inputr*inputWidth + inputc] * weights[i][j][r][c];
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
                                Dinputs[i*inputHeight*inputWidth + inputr*inputWidth + inputc] += weights[i][j][r][c] * Doutputs[j*outputHeight*outputWidth + x*outputWidth + y];
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

void ConvLayer::resetGradient(){
    int i,j,r,c;
    for(i=0; i<inputDepth; i++){
        for(j=0; j<outputDepth; j++){
            for(r=0; r<convHeight; r++){
                for(c=0; c<convWidth; c++){
                    Dweights[i][j][r][c] = 0;
                }
            }
        }
    }
    for(i=0; i<outputDepth; i++){
        Dbias[i] = 0;
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
                                Dweights[i][j][r][c] += inputs[i*inputHeight*inputWidth + inputr*inputWidth + inputc] * Dout;
                            }
                        }
                    }
                }
            }
        }
    }
}

void ConvLayer::updateParameters(double mult){
    int i,j,r,c;
    for(i=0; i<inputDepth; i++){
        for(j=0; j<outputDepth; j++){
            for(r=0; r<convHeight; r++){
                for(c=0; c<convWidth; c++){
                    weights[i][j][r][c] -= Dweights[i][j][r][c] * mult;
                    Dweights[i][j][r][c] *= momentum;
                }
            }
        }
    }
    for(i=0; i<outputDepth; i++){
        bias[i] -= Dbias[i] * mult;
        Dbias[i] *= momentum;
    }
}

void ConvLayer::save(){
    ofstream netOut(netAddress, ios::app);
    /*
    netOut<<"Input dimensions: "<<inputDepth<<" x "<<inputHeight<<" x "<<inputWidth<<'\n';
    netOut<<"Output dimensions: "<<outputDepth<<" x "<<outputHeight<<" x "<<outputWidth<<'\n';
    netOut<<"Conv dimensions: "<<convHeight<<" x "<<convWidth<<'\n';*/
    int i,j,r,c;
    for(i=0; i<inputDepth; i++){
        for(j=0; j<outputDepth; j++){
            for(r=0; r<convHeight; r++){
                for(c=0; c<convWidth; c++){
                    netOut << weights[i][j][r][c] << ' ';
                }
                netOut<<'\t';
            }
            netOut<<'\n';
        }
        netOut<<'\n';
    }
    for(i=0; i<outputDepth; i++){
        netOut << bias[i] << ' ';
    }
    netOut<<'\n';
    netOut.close();
}


// PoolLayer
    
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

void PoolLayer::save(){
    ofstream netOut(netAddress, ios::app);
    //netOut<<"Input dimensions: "<<inputDepth<<" x "<<inputHeight<<" x "<<inputWidth<<'\n';
    //netOut<<"Output dimensions: "<<outputDepth<<" x "<<outputHeight<<" x "<<outputWidth<<'\n';
    netOut.close();
}

// DenseLayer

void DenseLayer::randomize(){
    int i,j;
    for(i=0; i<inputSize; i++){
        for(j=0; j<outputSize; j++){
            weights[i][j] = randWeight();
        }
    }
    for(i=0; i<outputSize; i++){
        bias[i] = randWeight();
    }
}

void DenseLayer::pass(double* inputs, double* outputs){
    double sum;
    for(int i=0; i<outputSize; i++){
        sum = bias[i];
        for(int j=0; j<inputSize; j++){
            sum += weights[j][i] * inputs[j];
        }
        outputs[i] = nonlinear(sum);
    }
}

void DenseLayer::resetGradient(){
    int i,j;
    for(i=0; i<inputSize; i++){
        for(j=0; j<outputSize; j++){
            Dweights[i][j] = 0;
        }
    }
    for(i=0; i<outputSize; i++){
        Dbias[i] = 0;
    }
}

void DenseLayer::accumulateGradient(double* inputs, double* Doutputs){
    for(int i=0; i<outputSize; i++){
        Dbias[i] += Doutputs[i];
        for(int j=0; j<inputSize; j++){
            Dweights[j][i] += Doutputs[i] * inputs[j];
        }
    }
}

void DenseLayer::updateParameters(double mult){
    int i,j;
    for(i=0; i<inputSize; i++){
        for(j=0; j<outputSize; j++){
            weights[i][j] -= Dweights[i][j] * mult;
            Dweights[i][j] *= momentum;
        }
    }
    for(i=0; i<outputSize; i++){
        bias[i] -= Dbias[i] * mult;
        Dbias[i] *= momentum;
    }
}

void DenseLayer::backProp(double* inputs, double* Dinputs, double* Doutputs){
    double sum;
    for(int i=0; i<inputSize; i++){
        sum = 0;
        for(int j=0; j<outputSize; j++){
            sum += weights[i][j] * Doutputs[j];
        }
        Dinputs[i] = sum * dinvnonlinear(inputs[i]);
    }
}

void DenseLayer::save(){
    ofstream netOut(netAddress, ios::app);
    int i,j;
    for(i=0; i<inputSize; i++){
        for(j=0; j<outputSize; j++){
            netOut<<weights[i][j]<<' ';
        }
        netOut<<'\n';
    }
    netOut<<'\n';
    for(i=0; i<outputSize; i++){
        netOut<<bias[i]<<' ';
    }
    netOut<<"\n\n";
    netOut.close();
}


// Layer

void Layer::randomize(){
    if(type == 1){
        cl.initialize();
    }
    if(type == 2){
        // pass
    }
    if(type == 3){
        dl.randomize();
    }
}

void Layer::pass(double* inputs, double* outputs){
    if(type == 1){
        cl.pass(inputs, outputs);
    }
    if(type == 2){
        pl.pass(inputs, outputs);
    }
    if(type == 3){
        dl.pass(inputs, outputs);
    }
}

void Layer::backProp(double* inputs, double* Dinputs, double* Doutputs){
    if(type == 1){
        cl.backProp(inputs, Dinputs, Doutputs);
    }
    if(type == 2){
        pl.backProp(inputs, Dinputs, Doutputs);
    }
    if(type == 3){
        dl.backProp(inputs, Dinputs, Doutputs);
    }
}

void Layer::resetGradient(){
    if(type == 1){
        cl.resetGradient();
    }
    if(type == 2){
        // pass
    }
    if(type == 3){
        dl.resetGradient();
    }
}

void Layer::accumulateGradient(double* inputs, double* Doutputs){
    if(type == 1){
        cl.accumulateGradient(inputs, Doutputs);
    }
    if(type == 2){
        // pass
    }
    if(type == 3){
        dl.accumulateGradient(inputs, Doutputs);
    }
}

void Layer::updateParameters(double mult){
    if(type == 1){
        cl.updateParameters(mult);
    }
    if(type == 2){
        // pass
    }
    if(type == 3){
        dl.updateParameters(mult);
    }
}

void Layer::save(){
    if(type == 1){
        cl.save();
    }
    if(type == 2){
        pl.save();
    }
    if(type == 3){
        dl.save();
    }
}

// ConvNet

void Agent::initInput(int depth, int height, int width, int convHeight, int convWidth){
    il.outputDepth = depth;
    il.outputHeight = height;
    il.outputWidth = width;
    il.convHeight = convHeight;
    il.convWidth = convWidth;
    layerIndex = 0;
    prevDepth = depth;
    prevHeight = height;
    prevWidth = width;
}

void Agent::addConvLayer(int depth, int height, int width, int convHeight, int convWidth){
    layers[layerIndex].type = 1;
    layers[layerIndex].cl.inputDepth = prevDepth;
    layers[layerIndex].cl.inputHeight = prevHeight;
    layers[layerIndex].cl.inputWidth = prevWidth;
    layers[layerIndex].cl.outputDepth = depth;
    layers[layerIndex].cl.outputHeight = height;
    layers[layerIndex].cl.outputWidth = width;
    layers[layerIndex].cl.convHeight = convHeight;
    layers[layerIndex].cl.convWidth = convWidth;
    prevDepth = depth;
    prevHeight = height;
    prevWidth = width;
    layerIndex++;
}

void Agent::addPoolLayer(int depth, int height, int width){
    layers[layerIndex].type = 2;
    layers[layerIndex].pl.inputDepth = prevDepth;
    layers[layerIndex].pl.inputHeight = prevHeight;
    layers[layerIndex].pl.inputWidth = prevWidth;
    layers[layerIndex].pl.outputDepth = depth;
    layers[layerIndex].pl.outputHeight = height;
    layers[layerIndex].pl.outputWidth = width;
    prevDepth = depth;
    prevHeight = height;
    prevWidth = width;
    layerIndex++;
}

void Agent::addDenseLayer(int numNodes){
    layers[layerIndex].type = 3;
    layers[layerIndex].dl.inputSize = prevDepth * prevHeight * prevWidth;
    layers[layerIndex].dl.outputSize = numNodes;
    prevDepth = numNodes;
    prevHeight = 1;
    prevWidth = 1;
    layerIndex++;
}

void Agent::randomize(){
    il.initialize();
    for(int l=0; l<numLayers; l++){
        layers[l].randomize();
    }
}

void Agent::pass(){
    il.pass(&input, activation[0]);
    for(int l=0; l<numLayers; l++){
        layers[l].pass(activation[l], activation[l+1]);
    }
    output = activation[numLayers][0];
}

void Agent::resetGradient(){
    il.resetGradient();
    for(int l=0; l<numLayers; l++){
        layers[l].resetGradient();
    }
}

void Agent::backProp(){
    pass();
    Dbias[numLayers][0] = 2 * (activation[numLayers][0] - expected) * dinvnonlinear(activation[numLayers][0]);
    for(int l=numLayers-1; l>=0; l--){
        layers[l].accumulateGradient(activation[l], Dbias[l+1]);
        layers[l].backProp(activation[l], Dbias[l], Dbias[l+1]);
    }
    il.accumulateGradient(&input, Dbias[0]);
}

void Agent::updateParameters(double mult){
    il.updateParameters(mult);
    for(int l=0; l<numLayers; l++){
        layers[l].updateParameters(mult);
    }
}

void Agent::save(){
    il.save();
    for(int l=0; l<numLayers; l++){
        ofstream netOut(netAddress, ios::app);
        //netOut<<"Layer "<<l<<":\n";
        netOut.close();
        layers[l].save();
    }
}

