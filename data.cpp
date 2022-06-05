//
//  data.cpp
//  data
//
//  Created by Kevin Du on 2/2/22.
//

#include "snake.h"

Data::Data(Environment* givenEnv, double givenExpected){
    e.copyEnv(givenEnv);
    expectedValue = givenExpected;
}

void Data::trainAgent(Agent* a){
    e.inputSymmetric(a->input, rand()%8);
    a->expected = expectedValue;
    a->backProp();
}

DataQueue::DataQueue(){
    index = 0;
    for(int i=0; i<queueSize; i++){
        queue[i] = NULL;
    }
}

void DataQueue::enqueue(Data* d, int gameLength){
    if(queue[index%queueSize] != NULL){
        delete queue[index%queueSize];
    }
    queue[index%queueSize] = d;
    gameLengths[index % queueSize] = gameLength;
    index++;
}

void DataQueue::trainAgent(Agent* a){
    int i,j;
    for(i=0; i<numBatches; i++){
        for(j=0; j<batchSize; j++){
            int gameIndex = rand() % min(index,queueSize);
            queue[gameIndex][rand() % gameLengths[gameIndex]].trainAgent(a);
        }
        a->updateParameters(learnRate / batchSize, momentum);
    }
}
