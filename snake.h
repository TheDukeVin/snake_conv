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

#ifndef snake_h
#define snake_h

using namespace std;

//environment details

#define boardx 6
#define boardy 6
#define maxTime 180

#define numAgentActions 4
#define numChanceActions (boardx*boardy)
#define maxNumActions (boardx*boardy)

//training deatils

#define learnRate 0.01
#define momentum 0.9
#define batchSize 20
#define mult (learnRate / batchSize)
#define numEval 100
#define scoreNorm 5
#define numBatches 2
#define queueSize 40000

#define numGames 1501
#define numPaths 120
#define maxStates (maxTime*2*numPaths)
#define evalPeriod 100
#define numEvalGames 20
#define evalZscore 2

//network details

#define numLayers 4
#define maxNodes 100
#define maxDepth 4
#define maxConvSize 3
#define startingParameterRange 0.1


const string outAddress = "snake_conv.txt";
const string netAddress = "snakeConv_net.txt";

double squ(double x);

// Network things

double randWeight();
double nonlinear(double x);

//If f = nonlinear, then this function is (f' \circ f^{-1}).
double dinvnonlinear(double x);

class ConvLayer{
public:
    int inputDepth, inputHeight, inputWidth;
    int outputDepth, outputHeight, outputWidth;
    int convHeight, convWidth;
    int shiftr, shiftc;
    double weights[maxDepth][maxDepth][maxConvSize][maxConvSize]; // accessed in inputl, outputl, r, c.
    double bias[maxDepth];
    double Dweights[maxDepth][maxDepth][maxConvSize][maxConvSize];
    double Dbias[maxDepth];
    
    void initialize();
    void pass(double* inputs, double* outputs);
    void backProp(double* inputs, double* Dinputs, double* Doutputs);
    void resetGradient();
    void accumulateGradient(double* inputs, double* Doutputs);
    void updateParameters();
    void save();
};

class PoolLayer{
public:
    int inputDepth, inputHeight, inputWidth;
    int outputDepth, outputHeight, outputWidth;
    int maxIndices[maxNodes];
    
    void pass(double* inputs, double* outputs);
    void backProp(double* inputs, double* Dinputs, double* Doutputs);
    void save();
};

class DenseLayer{
public:
    int inputSize, outputSize;
    double weights[maxNodes][maxNodes];
    double bias[maxNodes];
    double Dweights[maxNodes][maxNodes];
    double Dbias[maxNodes];
    
    void randomize();
    void pass(double* inputs, double* outputs);
    void resetGradient();
    void accumulateGradient(double* inputs, double* Doutputs);
    void updateParameters();
    void backProp(double* inputs, double* Dinputs, double* Doutputs);
    void save();
};

class Layer{
public:
    int type;
    ConvLayer cl;
    PoolLayer pl;
    DenseLayer dl;
    
    void randomize();
    void pass(double* inputs, double* outputs);
    void backProp(double* inputs, double* Dinputs, double* Doutputs);
    void resetGradient();
    void accumulateGradient(double* inputs, double* Doutputs);
    void updateParameters();
    void save();
};


// Input layer tailored to snake environment

struct networkInput{
    int snake[boardx][boardy];
    int pos[3][2]; // head, tail, and apple positions
    int param[3]; // timer, score, and actionType
};

class InputLayer{
public:
    int outputDepth, outputHeight, outputWidth;
    int convHeight, convWidth;
    int shiftr, shiftc;
    double snakeWeights[4][maxDepth][maxConvSize][maxConvSize]; // accessed in cellType, outputl, r, c.
    double posWeights[3][maxDepth][maxConvSize][maxConvSize];
    double paramWeights[3][maxDepth];
    double bias[maxDepth];
    
    void initialize();
    void pass(networkInput* inputs, double* outputs);
    void resetGradient();
    void accumulateGradient(networkInput* inputs, double* Doutputs);
    void updateParameters();
    void save();
};

class Agent{
public:
    InputLayer il;
    networkInput input;
    Layer layers[numLayers];
    double activation[numLayers+1][maxNodes];
    double Dbias[numLayers+1][maxNodes]; // Dbias aligned with activation nodes.
    double output;
    double expected;
    
    // For network initiation
    int layerIndex;
    int prevDepth, prevHeight, prevWidth;
    
    void initInput(int depth, int height, int width, int convHeight, int convWidth);
    void addConvLayer(int depth, int height, int width, int convHeight, int convWidth);
    void addPoolLayer(int depth, int height, int width);
    void addDenseLayer(int numNodes);
    void randomize();
    
    // For network usage and training
    void pass();
    void resetGradient();
    void backProp();
    void updateParameters();
    void save();
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
    void inputAgent(Agent* a);
    void inputSymmetric(Agent* a, int t);
    void copyEnv(Environment* e);
    void print();// optional function for debugging
};

// Data things

class Data{
public:
    Environment e;
    double expectedValue;
    
    Data(Environment* givenEnv, double givenExpected){
        e.copyEnv(givenEnv);
        expectedValue = givenExpected;
    }
    
    void trainAgent(Agent* a){
        e.inputSymmetric(a, rand()%8);
        a->expected = expectedValue;
        a->backProp();
    }
};

class DataQueue{
public:
    Data* queue[queueSize];
    int index;
    
    DataQueue(){
        index = 0;
    }
    
    void enqueue(Data* d){
        queue[index%queueSize] = d;
        index++;
    }
    
    void trainAgent(Agent* a){
        int i,j;
        for(i=0; i<numBatches; i++){
            a->resetGradient();
            for(j=0; j<batchSize; j++){
                queue[rand() % min(index,queueSize)]->trainAgent(a);
            }
            a->updateParameters();
        }
    }
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
        a.randomize();
        a.resetGradient();
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
