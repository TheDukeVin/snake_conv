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
    e.inputSymmetric(&a->input, rand()%8);
    a->expected = expectedValue;
    a->backProp();
}

DataQueue::DataQueue(){
    index = 0;
    validLength = 1;
}

void DataQueue::enqueue(Data* d){
    queue[index] = d;
    index++;
    validLength = max(validLength, index);
    if(index == queueSize) index = 0;
}

void DataQueue::trainAgent(Agent* a){
    int i,j;
    for(i=0; i<numBatches; i++){
        for(j=0; j<batchSize; j++){
            queue[rand() % validLength]->trainAgent(a);
        }
        a->updateParameters(mult);
    }
}
