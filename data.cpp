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
    if(MODE == DETERMINISTIC_MODE){
        e.getDeterministicFeatures(features);
    }
}

void Data::trainAgent(Agent* a){
    e.inputSymmetric(a->input, rand()%8);
    a->expected = expectedValue;
    a->backProp();
    assert(abs(a->output) < 1000);
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

int DataQueue::readGames(){
    string input;
    ifstream fin("games.in");
    int maxScore = 0;
    while(fin >> input){
        if(input.length() <= 20){
            continue;
        }
        vector<int> actions;
        int currAction = 0;
        for(int i=0; i<input.length(); i++){
            if(input[i] == ','){
                actions.push_back(currAction);
                currAction = 0;
            }
            else{
                currAction *= 10;
                currAction += input[i] - '0';
            }
        }
        actions.push_back(currAction);
        int gameLength = actions.size();
        Data* game = new Data[gameLength];
        game[0].e.initialize();
        game[0].e.chanceAction(actions[0]);
        for(int i=1; i<gameLength; i++){
            assert(game[i-1].e.validAction(actions[i]));
            game[i].e.setAction(&game[i-1].e, actions[i]);
        }
        double value = game[gameLength-1].e.getReward();
        for(int i=gameLength-1; i>=0; i--){
            game[i].expectedValue = value;
            if(i > 0){
                value = game[i-1].e.getReward() + value * pow(discountFactor, game[i].e.timer - game[i-1].e.timer);
            }
        }
        enqueue(game, gameLength);
        //maxScore = max(maxScore, game[gameLength - 1].e.getScore());
        cout<<game[gameLength - 1].e.snakeSize<<',';
    }
    cout<<"\n\n";
    return maxScore;
}

void DataQueue::trainLinear(LinearModel* lm){
    for(int i=0; i<batchSize; i++){
        int gameIndex = rand() % min(index,queueSize);
        Data* chosenData = &(queue[gameIndex][rand() % gameLengths[gameIndex]]);
        lm->backProp(chosenData->features, chosenData->expectedValue);
    }
    lm->updateParameters(learnRate / batchSize, momentum);
}
