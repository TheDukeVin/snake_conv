//
//  snake.h
//  snake
//
//  Created by Kevin Du on 1/18/22.
//

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <string>
#include <vector>
#include <list>

#ifndef snake_h
#define snake_h

using namespace std;

//environment details

#define boardx 10
#define boardy 10
#define maxTime 1200

#define numAgentActions 4
#define numChanceActions (boardx*boardy)
#define maxNumActions (boardx*boardy)

//training deatils

#define maxNorm 100
#define batchSize 100
#define numBatches 30

//#define scoreNorm 10
#define queueSize 500

#define numGames 4000
#define numPaths 200
#define explorationConstant 0.3

#define maxStates (maxTime*2*numPaths)
#define evalPeriod 100
#define numEvalGames 100
#define evalZscore 2

#define discountFactor 0.95

// Deterministic vs Network mode

#define DETERMINISTIC_MODE 0
#define NETWORK_MODE 1
#define MODE NETWORK_MODE
#define numFeatures 8

const string outAddress = "snake_conv.txt";

double squ(double x);

int max(int x, int y);

double min(double x, double y);

// For the network
double randWeight(double startingParameterRange);

double nonlinear(double x);

// If f = nonlinear, then this function is (f' \circ f^{-1}).
double dinvnonlinear(double x);

// Input to the network

struct networkInput{
    int snake[boardx][boardy];
    int pos[3][2]; // head, tail, and apple positions
    //double param[3]; // timer, score, and actionType. score and timer are normalized.
};

class Layer{
public:
    ifstream* netIn;
    ofstream* netOut;
    int numParams, numWeights, numBias;
    double* params;
    double* weights;
    double* bias;
    double* Dparams;
    double* Dweights;
    double* Dbias;
    virtual void pass(double* inputs, double* outputs){};
    virtual void backProp(double* inputs, double* Dinputs, double* Doutputs){};
    virtual void accumulateGradient(double* inputs, double* Doutputs){};
    void setupParams();
    void randomize(double startingParameterRange);
    void resetGradient();
    void updateParameters(double mult, double momentum);
    void save();
    void readNet();
    
    virtual ~Layer(){}
};

class ConvLayer : public Layer{
public:
    int inputDepth, inputHeight, inputWidth;
    int outputDepth, outputHeight, outputWidth;
    int convHeight, convWidth;
    int shiftr, shiftc;
    int w1, w2, w3;
    /*
    double weights[maxDepth][maxDepth][maxConvSize][maxConvSize]; // accessed in inputl, outputl, r, c.
    double bias[maxDepth];
    double Dweights[maxDepth][maxDepth][maxConvSize][maxConvSize];
    double Dbias[maxDepth];*/
    
    ConvLayer(int inD, int inH, int inW, int outD, int outH, int outW, int convH, int convW);
    
    virtual void pass(double* inputs, double* outputs);
    virtual void backProp(double* inputs, double* Dinputs, double* Doutputs);
    virtual void accumulateGradient(double* inputs, double* Doutputs);
    
    virtual ~ConvLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

class PoolLayer : public Layer{
public:
    int inputDepth, inputHeight, inputWidth;
    int outputDepth, outputHeight, outputWidth;
    int* maxIndices;
    
    PoolLayer(int inD, int inH, int inW, int outD, int outH, int outW);
    
    virtual void pass(double* inputs, double* outputs);
    virtual void backProp(double* inputs, double* Dinputs, double* Doutputs);
    
    virtual ~PoolLayer(){
        delete[] maxIndices;
    }
};

class DenseLayer : public Layer{
public:
    int inputSize, outputSize;
    
    DenseLayer(int inSize, int outSize);
    
    virtual void pass(double* inputs, double* outputs);
    virtual void backProp(double* inputs, double* Dinputs, double* Doutputs);
    virtual void accumulateGradient(double* inputs, double* Doutputs);
    
    virtual ~DenseLayer(){
        delete[] params;
        delete[] Dparams;
    }
};


// Input layer tailored to snake environment

class InputLayer : public Layer{
public:
    int outputDepth, outputHeight, outputWidth;
    int convHeight, convWidth;
    int shiftr, shiftc;
    int posShiftr, posShiftc;
    int w1, w2, w3;
    
    networkInput* env;
    
    double* snakeWeights; // accessed in cellType, outputl, r, c.
    double* posWeights;
    //double* paramWeights;
    
    double* DsnakeWeights;
    double* DposWeights;
    //double* DparamWeights;
    
    InputLayer(int outD, int outH, int outW, int convH, int convW, networkInput* input);
    
    virtual void pass(double* inputs, double* outputs);
    virtual void accumulateGradient(double* inputs, double* Doutputs);
    
    virtual ~InputLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

class OutputLayer : public Layer{
public:
    int inputSize, outputSize; // outputSize = 1.
    
    OutputLayer(int inSize, int outputSize);
    
    virtual void pass(double* inputs, double* outputs);
    virtual void backProp(double* inputs, double* Dinputs, double* Doutputs);
    virtual void accumulateGradient(double* inputs, double* Doutputs);
    
    virtual ~OutputLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

class Agent{
public:
    networkInput* input;
    unsigned long numLayers;
    unsigned maxNodes = 0;
    Layer** layers; // keep an array of pointers, since derived classes need to be accessed by reference.
    double** activation;
    double** Dbias;
    
    double output;
    double expected;
    
    // For file I/O
    ifstream* netIn;
    ofstream* netOut;
    
    void initInput(int depth, int height, int width, int convHeight, int convWidth);
    void addConvLayer(int depth, int height, int width, int convHeight, int convWidth);
    void addPoolLayer(int depth, int height, int width);
    void addDenseLayer(int numNodes);
    void addOutputLayer();
    void randomize(double startingParameterRange);
    
    // For network usage and training
    void pass();
    void resetGradient();
    void backProp();
    void updateParameters(double mult, double momentum);
    void save(string fileName);
    void readNet(string fileName);
    
private:
    // For network initiation
    int prevDepth, prevHeight, prevWidth;
    vector<Layer*> layerHold;
};

class LinearModel{
    public:
    int numParams;
    double* params;
    double* weights;
    double* bias;
    double* Dparams;
    double* Dweights;
    double* Dbias;

    LinearModel(){
        numParams = numFeatures + 1;
        params = new double[numParams];
        weights = params;
        bias = params + numFeatures;
        Dparams = new double[numParams];
        Dweights = Dparams;
        Dbias = Dparams + numFeatures;

        // Initial set of weights:
        for(int i=0; i<numParams; i++){
            params[i] = 0;
        }
        //-0.357594  1.29193 0.0270724       0.15542 0.162815        0.855346        0.0896465       0.00498049      0.877094
        /*
        weights[0] = -0.357594;
        weights[1] = 1.29193;
        weights[2] = 0.0270724;
        weights[3] = 0.15542;
        weights[4] = 0.162815;
        weights[5] = 0.855346;
        weights[6] = 0.0896465;
        weights[7] = 0.00498049;
        bias[0] = 0.877094;
        */
        resetGradient();
    }

    double pass(double* features);

    void backProp(double* features, double expected);

    void resetGradient();
    void updateParameters(double mult, double momentum);
};

// Environment things

const int numActions[2] = {numAgentActions, numChanceActions};

const int dir[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};

class Environment{
private:
    double score;
public:
    int timer;
    int actionType; // 0 = action state, 1 = reaction state.
    
    int snakeSize;
    int headx,heady;
    int tailx, taily;
    int applex,appley;
    int snake[boardx][boardy]; // -1 = not snake. 0 to 3 = snake unit pointing to next unit. 4 = head.
    
    void initialize();
    
    bool isEndState();
    bool validAction(int actionIndex); // returns whether the action is valid.
    bool validAgentAction(int d);
    bool validChanceAction(int pos);
    void makeAction(int actionIndex);
    void setAction(Environment* currState, int actionIndex);
    void inputSymmetric(networkInput* a, int t);
    void copyEnv(Environment* e);
    void print();// optional function for debugging
    void log();// optional function for debugging
    
    void getDeterministicFeatures(double* features);
    double getReward();
    
    void agentAction(int actionIndex);
    void chanceAction(int actionIndex);
    
private:
    double getScore();
};

// Data things

class Data{
public:
    Environment e;
    double expectedValue;
    double features[numFeatures];
    
    Data(){}
    Data(Environment* givenEnv, double givenExpected);
    void trainAgent(Agent* a);
};

class DataQueue{
public:
    Data* queue[queueSize];
    int gameLengths[queueSize];
    int index;
    double learnRate, momentum;
    
    DataQueue();
    void enqueue(Data* d, int gameLength);
    void trainAgent(Agent* a);
    vector<int> readGames(); // returns the maximum score out of the games read.

    void trainLinear(LinearModel* lm);
};

// Trainer

class Trainer{
public:
    DataQueue* dq;
    
    double actionTemperature = 1;
    
    Agent a;
    LinearModel lm;
    double exploitationFactor;
    
    string gameLog;
    string valueLog;
    
    Trainer(DataQueue* givendq){
        dq = givendq;
        exploitationFactor = 1;
        for(int i=0; i<maxStates; i++){
            outcomes[i] = NULL;
        }
    }
    
    //Storage for the tree:
    int* outcomes[maxStates];
    int subtreeSize[maxStates];
    double sumScore[maxStates];
    Environment roots[maxTime*2];
    
    // Implementing the tree search
    int index;
    int rootIndex, rootState;
    
    // For executing a training iteration:
    double actionProbs[numAgentActions];
    
    void initializeNode(Environment& env, int currNode);
    double trainTree();
    int evalGame();// return index of the final state in states.
    void printGame();
    void exportGame();
    void evaluate();
    
    int path[maxStates];
    double rewards[maxTime*2];
    int times[maxTime*2];
    
    void expandPath();
    void printTree();
    void computeActionProbs();
    int optActionProbs();
    int sampleActionProbs();
    int getRandomChanceAction(Environment* e);
};

#endif /* snake_h */
