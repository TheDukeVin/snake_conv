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

#ifndef snake_h
#define snake_h

using namespace std;

//environment details

#define boardx 6
#define boardy 6
#define maxTime 80

#define numAgentActions 4
#define numChanceActions (boardx*boardy)
#define maxNumActions (boardx*boardy)

//training deatils

#define learnRate 0.05
#define momentum 0
#define maxNorm 1
#define batchSize 2000
#define mult (learnRate / batchSize)

#define scoreNorm 5
#define numBatches 1
#define queueSize 12000

#define numGames 1501
#define numPaths 120
#define maxStates (maxTime*2*numPaths)
#define evalPeriod 100
#define numEvalGames 20
#define evalZscore 2

const string outAddress = "snake_conv.txt";

double squ(double x);

// For the network
double randWeight(double startingParameterRange);

double nonlinear(double x);

// If f = nonlinear, then this function is (f' \circ f^{-1}).
double dinvnonlinear(double x);

// Input to the network

struct networkInput{
    int snake[boardx][boardy];
    int pos[3][2]; // head, tail, and apple positions
    double param[3]; // timer, score, and actionType. score and timer are normalized.
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
    void updateParameters();
    void save();
    void readNet();
    
    virtual ~Layer(){}
    virtual void setInput(networkInput* _env){}
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
    virtual void setInput(networkInput* _env){
        env = _env;
    }
    
    double* snakeWeights; // accessed in cellType, outputl, r, c.
    double* posWeights;
    double* paramWeights;
    
    double* DsnakeWeights;
    double* DposWeights;
    double* DparamWeights;
    
    InputLayer(int outD, int outH, int outW, int convH, int convW, networkInput* input);
    
    virtual void pass(double* inputs, double* outputs);
    virtual void accumulateGradient(double* inputs, double* Doutputs);
    
    virtual ~InputLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

class Agent{
public:
    networkInput* input;
    unsigned long numLayers;
    unsigned maxNodes = 0;
    Layer** layers; // keep an array of pointers, since abstract classes need to be accessed by reference.
    double** activation;
    double** Dbias;
    double output;
    double expected;
    
    // For file I/O
    ifstream netIn;
    ofstream netOut;
    
    void initInput(int depth, int height, int width, int convHeight, int convWidth);
    void addConvLayer(int depth, int height, int width, int convHeight, int convWidth);
    void addPoolLayer(int depth, int height, int width);
    void addDenseLayer(int numNodes);
    void randomize(double startingParameterRange);
    
    // For network usage and training
    void quickSetup();
    void pass();
    void resetGradient();
    void backProp();
    void updateParameters();
    void save();
    void readNet();
    
private:
    // For network initiation
    int prevDepth, prevHeight, prevWidth;
    vector<Layer*> layerHold;
    void setupIO();
};


// Environment things

const int numActions[2] = {numAgentActions, numChanceActions};

const int dir[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};

class Environment{
public:
    int timer;
    double score;
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
    void setAction(Environment* currState, int actionIndex);
    void setAgentAction(Environment* currState, int actionIndex);
    void setChanceAction(Environment* currState, int actionIndex);
    void inputSymmetric(networkInput* a, int t);
    void copyEnv(Environment* e);
    void print();// optional function for debugging
};

// Data things

class Data{
public:
    Environment e;
    double expectedValue;
    
    Data(Environment* givenEnv, double givenExpected);
    void trainAgent(Agent* a);
};

class DataQueue{
public:
    Data* queue[queueSize];
    int index;
    
    DataQueue();
    void enqueue(Data* d);
    void trainAgent(Agent* a);
};

// Trainer


class Trainer{
public:
    Environment* states;
    DataQueue* dq;
    
    Agent a;
    double exploitationFactor;
    
    Trainer(Environment* givenStates, DataQueue* givendq){
        states = givenStates;
        dq = givendq;
        exploitationFactor = 1;
    }
    
    //Storage for the tree:
    int outcomes[maxStates][maxNumActions];
    int size[maxStates];
    double sumScore[maxStates];
    int index;
    
    //For executing a training iteration:
    int roots[maxStates];
    int currRoot;
    double actionProbs[numAgentActions];
    
    void initializeNode(int currNode);
    void trainTree();
    int evalGame();// return index of the final state in states.
    void printGame();
    double evaluate();
    
    int path[maxStates];
    
    void expandPath();
    void printTree();
    void computeActionProbs();
    int optActionProbs();
    int sampleActionProbs();
    int getRandomChanceAction(Environment* e);
};

#endif /* snake_h */
