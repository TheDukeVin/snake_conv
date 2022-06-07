//
//  trainer.cpp
//  trainer
//
//  Created by Kevin Du on 1/18/22.
//

#include "snake.h"



void Trainer::initializeNode(int currNode){
    for(int i=0; i<numActions[states[currNode].actionType]; i++){
        if(!states[currNode].validAction(i)){
            outcomes[currNode][i] = -2;
        }
        else{
            outcomes[currNode][i] = -1;
        }
    }
    size[currNode] = 0;
    sumScore[currNode] = 0;
}

double Trainer::trainTree(){
    states[0].initialize();
    ofstream fout(gameLog, ios::app);
    fout<<(states[0].applex * boardy + states[0].appley)<<',';
    initializeNode(0);
    currRoot = 0;
    roots[0] = 0;
    index = 1;
    
    int s = 0;
    int chosenAction;
    int i;
    while(!states[currRoot].isEndState()){
        if(states[currRoot].actionType == 0){
            for(i=0; i<numPaths; i++){
                expandPath();
            }
            computeActionProbs();
            chosenAction = sampleActionProbs();
        }
        if(states[currRoot].actionType == 1){
            chosenAction = getRandomChanceAction(&states[currRoot]);
            if(outcomes[currRoot][chosenAction] == -1){
                outcomes[currRoot][chosenAction] = index;
                states[index].setAction(&states[currRoot], chosenAction);
                initializeNode(index);
                index++;
            }
        }
        currRoot = outcomes[currRoot][chosenAction];
        if(states[currRoot].isEndState()){
            fout<<chosenAction;
        }
        else{
            fout<<chosenAction<<',';
        }
        s++;
        roots[s] = currRoot;
    }
    fout<<'\n';
    fout.close();
    int numStates = s;
    Data* game = new Data[numStates];
    double finalScore = states[currRoot].getScore();
    for(i=0; i<numStates; i++){
        game[i] = Data(&states[roots[i]], finalScore);
    }
    dq->enqueue(game, numStates);
    return finalScore;
}

int Trainer::evalGame(){ // return index of the final state in states.
    states[0].initialize();
    initializeNode(0);
    currRoot = 0;
    roots[0] = 0;
    index = 1;
    
    int s = 0;
    int chosenAction;
    int i;
    while(!states[currRoot].isEndState()){
        if(states[currRoot].actionType == 0){
            for(i=0; i<numPaths; i++){
                expandPath();
            }
            computeActionProbs();
            chosenAction = sampleActionProbs();
        }
        if(states[currRoot].actionType == 1){
            chosenAction = getRandomChanceAction(&states[currRoot]);
            if(outcomes[currRoot][chosenAction] == -1){
                outcomes[currRoot][chosenAction] = index;
                states[index].setAction(&states[currRoot], chosenAction);
                initializeNode(index);
                index++;
            }
        }
        currRoot = outcomes[currRoot][chosenAction];
        s++;
        roots[s] = currRoot;
    }
    return currRoot;
}

void Trainer::printGame(){
    states[0].initialize();
    initializeNode(0);
    currRoot = 0;
    roots[0] = 0;
    index = 1;
    
    int s = 0;
    int chosenAction;
    int i;
    while(!states[currRoot].isEndState()){
        states[currRoot].print();
        if(states[currRoot].actionType == 0){
            for(i=0; i<numPaths; i++){
                expandPath();
            }
            computeActionProbs();
            ofstream fout(outAddress, ios::app);
            fout<<"Action probabilities: ";
            for(i=0; i<numAgentActions; i++){
                fout<<actionProbs[i]<<' ';
            }
            fout<<"\n\n";
            fout.close();
            chosenAction = optActionProbs();
        }
        if(states[currRoot].actionType == 1){
            chosenAction = getRandomChanceAction(&states[currRoot]);
            if(outcomes[currRoot][chosenAction] == -1){
                outcomes[currRoot][chosenAction] = index;
                states[index].setAction(&states[currRoot], chosenAction);
                initializeNode(index);
                index++;
            }
        }
        currRoot = outcomes[currRoot][chosenAction];
        s++;
        roots[s] = currRoot;
    }
}

void Trainer::exportGame(){
    states[0].initialize();
    cout<<(states[0].applex * boardy + states[0].appley)<<',';
    initializeNode(0);
    currRoot = 0;
    roots[0] = 0;
    index = 1;
    
    int s = 0;
    int chosenAction;
    int i;
    while(!states[currRoot].isEndState()){
        if(states[currRoot].actionType == 0){
            for(i=0; i<numPaths; i++){
                expandPath();
            }
            computeActionProbs();
            chosenAction = optActionProbs();
        }
        if(states[currRoot].actionType == 1){
            chosenAction = getRandomChanceAction(&states[currRoot]);
            if(outcomes[currRoot][chosenAction] == -1){
                outcomes[currRoot][chosenAction] = index;
                states[index].setAction(&states[currRoot], chosenAction);
                initializeNode(index);
                index++;
            }
        }
        currRoot = outcomes[currRoot][chosenAction];
        s++;
        roots[s] = currRoot;
        if(states[currRoot].isEndState()){
            cout<<chosenAction<<'\n';
        }
        else{
            cout<<chosenAction<<',';
        }
    }
}


double Trainer::evaluate(){
    int endState;
    double scoreSum = 0;
    double sizeSum = 0;
    double scoreSquareSum = 0;
    int numCompletes = 0;
    int timeSum = 0;
    for(int i=0; i<numEvalGames; i++){
        endState = evalGame();
        cout<<states[endState].getScore()<<' ';
        scoreSum += states[endState].getScore();
        sizeSum += states[endState].snakeSize;
        scoreSquareSum += squ(states[endState].getScore());
        if(states[endState].snakeSize == boardx * boardy){
            numCompletes++;
            timeSum += states[endState].timer;
        }
    }
    double averageScore = scoreSum / numEvalGames;
    double variance = scoreSquareSum / numEvalGames - squ(averageScore);
    double SE = sqrt(variance / numEvalGames) * evalZscore;
    cout<<"\nAverage snake size: "<<(sizeSum/numEvalGames)<<'\n';
    cout<<"Average score: "<<averageScore<<'\n';
    cout<<"Confidence interval: (" << (averageScore - SE) << ", " << (averageScore + SE) << ")\n";
    cout<<"Proportion of completions: "<<((double) numCompletes / numEvalGames)<<'\n';
    cout<<"Average time to completion: "<<((double) timeSum / numCompletes)<<'\n';
    cout<<'\n';
    return averageScore;
}

void Trainer::expandPath(){
    int currNode = currRoot;
    int nextNode,nextAction;
    int count = 0;
    int currType;
    int maxIndex;
    double maxVal,candVal;
    int i;
    while(currNode != -1 && !states[currNode].isEndState()){
        path[count] = currNode;
        count++;
        maxVal = -1000000;
        currType = states[currNode].actionType;
        for(i=0; i<numActions[currType]; i++){
            nextNode = outcomes[currNode][i];
            if(nextNode == -2){
                continue;
            }
            if(nextNode == -1){
                candVal = 1000 + (double)rand() / RAND_MAX;
            }
            else{
                if(currType == 0){
                    candVal = sumScore[nextNode] / size[nextNode] + 0.5 * log(size[currNode]) / sqrt(size[nextNode]);
                }
                if(currType == 1){
                    candVal = (double)rand() / RAND_MAX - size[nextNode];
                }
            }
            if(candVal > maxVal){
                maxVal = candVal;
                maxIndex = i;
            }
        }
        nextAction = maxIndex;
        currNode = outcomes[currNode][maxIndex];
    }
    double newVal;
    if(currNode == -1){
        outcomes[path[count-1]][nextAction] = index;
        states[index].setAction(&states[path[count-1]], nextAction);
        initializeNode(index);
        if(hard_code){
            newVal = states[index].snakeSize - states[index].actionType;
            newVal += (maxTime - states[index].timer) * 0.05;
            newVal -= (abs(states[index].headx - states[index].applex)) * 0.05;
            newVal -= (abs(states[index].heady - states[index].appley)) * 0.05;
        }
        else{
            states[index].inputSymmetric(a.input, rand()%8);
            a.pass();
            newVal = a.output;
        }
        
        assert(abs(newVal) < 1000);
        path[count] = index;
        index++;
        count++;
        /*
        Attempt at voiding very bad scores. Crashes.
        if(size[currRoot] > 0 && newVal < (sumScore[currRoot] / size[currRoot]) - 2){
            cout<<sumScore[currRoot] <<' '<< size[currRoot]<<'\n';
            outcomes[path[count-1]][nextAction] = -2;
            return;
        }
        else{
            outcomes[path[count-1]][nextAction] = index;
            path[count] = index;
            index++;
            count++;
        }
        */
    }
    else{
        newVal = states[currNode].getScore();
        path[count] = currNode;
        count++;
    }
    for(i=0; i<count; i++){
        size[path[i]]++;
        sumScore[path[i]] += newVal;
    }
}

void Trainer::printTree(){
    ofstream fout(outAddress, ios::app);
    for(int i=0; i<index; i++){
        fout<<"State "<<i<<'\n';
        states[i].print();
        fout<<"Outcomes: ";
        for(int j=0; j<numActions[states[i].actionType]; j++){
            fout<<outcomes[i][j];
        }
        fout<<'\n';
        fout<<"Size: "<<size[i]<<'\n';
        fout<<"Sum score: "<<sumScore[i]<<'\n';
        fout<<'\n';
    }
    fout.close();
}

void Trainer::computeActionProbs(){
    int i;
    int nextIndex;
    for(i=0; i<numAgentActions; i++){
        nextIndex = outcomes[currRoot][i];
        if(nextIndex != -2){
            actionProbs[i] = pow(size[nextIndex], 2); // squ(size[nextIndex]);
        }
        else{
            actionProbs[i] = -1;
        }
    }
}

int Trainer::optActionProbs(){
    int i;
    int maxIndex = 0;
    for(i=1; i<numAgentActions; i++){
        if(actionProbs[i] > actionProbs[maxIndex]){
            maxIndex = i;
        }
    }
    return maxIndex;
}

int Trainer::sampleActionProbs(){
    double propSum = 0;
    int i;
    for(i=0; i<numAgentActions; i++){
        if(actionProbs[i] == -1){
            continue;
        }
        propSum += actionProbs[i];
    }
    double parsum = 0;
    double randReal = (double)rand() / RAND_MAX * propSum;
    
    int actionIndex = -1;
    for(i=0; i<numAgentActions; i++){
        if(actionProbs[i] == -1){
            continue;
        }
        parsum += actionProbs[i];
        if(randReal <= parsum){
            actionIndex = i;
            break;
        }
    }
    return actionIndex;
}

int Trainer::getRandomChanceAction(Environment* e){
    int i;
    int possibleActions[numChanceActions];
    int numPossibleActions = 0;
    for(i=0; i<numChanceActions; i++){
        if(e->validAction(i)){
            possibleActions[numPossibleActions] = i;
            numPossibleActions++;
        }
    }
    return possibleActions[rand() % numPossibleActions];
}

