//
//  InputLayerCode.cpp
//  InputLayerCode
//
//  Created by Kevin Du on 1/22/22.
//

#include "snake.h"

void InputLayer::initialize(){
    shiftr = (boardx - outputHeight - convHeight + 1) / 2;
    shiftc = (boardy - outputWidth - convWidth + 1) / 2;
    posShiftr = (1 - convHeight) / 2;
    posShiftc = (1 - convWidth) / 2;
    int i,j,r,c;
    for(j=0; j<outputDepth; j++){
        for(r=0; r<convHeight; r++){
            for(c=0; c<convWidth; c++){
                for(i=0; i<4; i++){
                    snakeWeights[i][j][r][c] = randWeight();
                }
                for(i=0; i<3; i++){
                    posWeights[i][j][r][c] = randWeight();
                }
            }
        }
    }
    for(i=0; i<3; i++){
        for(j=0; j<outputDepth; j++){
            paramWeights[i][j] = randWeight();
        }
    }
    for(i=0; i<outputDepth; i++){
        bias[i] = randWeight();
    }
}

void InputLayer::pass(networkInput* inputs, double* outputs){
    double inc;
    for(int j=0; j<outputDepth; j++){
        inc = 0;
        for(int i=0; i<3; i++){
            inc += inputs->param[i] * paramWeights[i][j];
        }
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                outputs[j*outputHeight*outputWidth + x*outputWidth + y] = bias[j] + inc;
            }
        }
    }
    for(int i=0; i<3; i++){
        for(int r=0; r<convHeight; r++){
            for(int c=0; c<convWidth; c++){
                int outputr = inputs->pos[i][0] + r + posShiftr;
                int outputc = inputs->pos[i][1] + c + posShiftc;
                if(outputr >= 0 && outputr < outputHeight && outputc >= 0 && outputc < outputWidth){
                    for(int j=0; j<outputDepth; j++){
                        outputs[j*outputHeight*outputWidth + outputr*outputWidth + outputc] += posWeights[i][j][r][c];
                    }
                }
            }
        }
    }
    double output;
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                output = outputs[j*outputHeight*outputWidth + x*outputWidth + y];
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < boardx && inputc >= 0 && inputc < boardy){
                            int input = inputs->snake[inputr][inputc];
                            if(input != -1){
                                output += snakeWeights[input][j][r][c];
                            }
                        }
                    }
                }
                outputs[j*outputHeight*outputWidth + x*outputWidth + y] = nonlinear(output);
            }
        }
    }
}

void InputLayer::resetGradient(){
    int i,j,r,c;
    for(j=0; j<outputDepth; j++){
        for(r=0; r<convHeight; r++){
            for(c=0; c<convWidth; c++){
                for(i=0; i<4; i++){
                    DsnakeWeights[i][j][r][c] = 0;
                }
                for(i=0; i<3; i++){
                    DposWeights[i][j][r][c] = 0;
                }
            }
        }
    }
    for(i=0; i<3; i++){
        for(j=0; j<outputDepth; j++){
            DparamWeights[i][j] = 0;
        }
    }
    for(i=0; i<outputDepth; i++){
        Dbias[i] = 0;
    }
}

void InputLayer::accumulateGradient(networkInput* inputs, double* Doutputs){
    double sum;
    for(int j=0; j<outputDepth; j++){
        sum = 0;
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                sum += Doutputs[j*outputHeight*outputWidth + x*outputWidth + y];
            }
        }
        for(int i=0; i<3; i++){
            DparamWeights[i][j] += sum * inputs->param[i];
        }
        Dbias[j] += sum;
    }
    for(int i=0; i<3; i++){
        for(int r=0; r<convHeight; r++){
            for(int c=0; c<convWidth; c++){
                int outputr = inputs->pos[i][0] + r + posShiftr;
                int outputc = inputs->pos[i][1] + c + posShiftc;
                if(outputr >= 0 && outputr < outputHeight && outputc >= 0 && outputc < outputWidth){
                    for(int j=0; j<outputDepth; j++){
                        DposWeights[i][j][r][c] += Doutputs[j*outputHeight*outputWidth + outputr*outputWidth + outputc];
                    }
                }
            }
        }
    }
    double Doutput;
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                Doutput = Doutputs[j*outputHeight*outputWidth + x*outputWidth + y];
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < boardx && inputc >= 0 && inputc < boardy){
                            int input = inputs->snake[inputr][inputc];
                            if(input != -1){
                                DsnakeWeights[input][j][r][c] += Doutput;
                            }
                        }
                    }
                }
            }
        }
    }
}

void InputLayer::updateParameters(){
    int i,j,r,c;
    for(j=0; j<outputDepth; j++){
        for(r=0; r<convHeight; r++){
            for(c=0; c<convWidth; c++){
                for(i=0; i<4; i++){
                    snakeWeights[i][j][r][c] -= DsnakeWeights[i][j][r][c] * mult;
                }
                for(i=0; i<3; i++){
                    posWeights[i][j][r][c] -= DposWeights[i][j][r][c] * mult;
                }
            }
        }
    }
    for(i=0; i<3; i++){
        for(j=0; j<outputDepth; j++){
            paramWeights[i][j] -= DparamWeights[i][j] * mult;
        }
    }
    for(i=0; i<outputDepth; i++){
        bias[i] -= Dbias[i] * mult;
    }
}

void InputLayer::save(){
    ofstream netOut(netAddress, ios::app);
    netOut<<"Output dimensions: "<<outputDepth<<" x "<<outputHeight<<" x "<<outputWidth<<'\n';
    netOut<<"Conv dimensions: "<<convHeight<<" x "<<convWidth<<'\n';
    int i,j,r,c;
    netOut<<"Snake Weights:\n";
    for(i=0; i<4; i++){
        for(j=0; j<outputDepth; j++){
            for(r=0; r<convHeight; r++){
                for(c=0; c<convWidth; c++){
                    netOut << snakeWeights[i][j][r][c] << ' ';
                }
                netOut<<'\t';
            }
            netOut<<'\n';
        }
        netOut<<'\n';
    }
    netOut<<"\nPos Weights:\n";
    for(i=0; i<3; i++){
        for(j=0; j<outputDepth; j++){
            for(r=0; r<convHeight; r++){
                for(c=0; c<convWidth; c++){
                    netOut << posWeights[i][j][r][c] << ' ';
                }
                netOut<<'\t';
            }
            netOut<<'\n';
        }
        netOut<<'\n';
    }
    netOut<<"\nParam Weights:\n";
    for(i=0; i<3; i++){
        for(j=0; j<outputDepth; j++){
            netOut << paramWeights[i][j] << ' ';
        }
        netOut<<'\n';
    }
    netOut<<"\nBias:\n";
    for(i=0; i<outputDepth; i++){
        netOut << bias[i] << ' ';
    }
    netOut<<'\n';
    netOut.close();
}
