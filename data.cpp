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
    int orient = 0;
    e.inputSymmetric(a->input, orient);
    a->expected[0] = expectedValue;
    for(int i=0; i<4; i++){
        int newDir = (symDir[orient][0]*i + symDir[orient][1] + 4) % 4;
        a->expected[newDir+1] = policy[i];
    }
    a->backProp();
}

DataQueue::DataQueue(){
    index = 0;
}

void DataQueue::enqueue(Data* d){
    if(queue[index%queueSize]) delete queue[index%queueSize];
    queue[index%queueSize] = d;
    index++;
}

void DataQueue::trainAgent(Agent* a){
    int i,j;
    for(i=0; i<numBatches; i++){
        for(j=0; j<batchSize; j++){
            queue[rand() % min(index,queueSize)]->trainAgent(a);
        }
        a->updateParameters(learnRate / batchSize, momentum);
    }
}
