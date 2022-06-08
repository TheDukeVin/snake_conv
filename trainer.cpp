//
//  trainer.cpp
//  trainer
//
//  Created by Kevin Du on 1/18/22.
//

#include "snake.h"



void Trainer::initializeNode(Environment& env, int currNode){
    if(outcomes[currNode] != NULL){
        delete outcomes[currNode];
    }
    int numOutcomes = numActions[env.actionType];
    outcomes[currNode] = new int[numOutcomes];
    for(int i=0; i<numOutcomes; i++){
        if(!env.validAction(i)){
            outcomes[currNode][i] = -2;
        }
        else{
            outcomes[currNode][i] = -1;
        }
    }
    subtreeSize[currNode] = 0;
    sumScore[currNode] = 0;
}

double Trainer::trainTree(){
    roots[0].initialize();
    rootIndex = 0;
    ofstream fout(gameLog, ios::app);
    fout<<(roots[0].applex * boardy + roots[0].appley)<<',';
    initializeNode(roots[0], 0);
    index = 1;
    
    int chosenAction;
    for(rootState=0; rootState<maxTime*2; rootState++){
        if(roots[rootState].actionType == 0){
            for(int j=0; j<numPaths; j++){
                expandPath();
            }
            computeActionProbs();
            chosenAction = sampleActionProbs();
        }
        else{
            chosenAction = getRandomChanceAction(&roots[rootState]);
            if(outcomes[rootIndex][chosenAction] == -1){
                outcomes[rootIndex][chosenAction] = index;
                Environment env;
                env.setAction(&roots[rootState], chosenAction);
                initializeNode(env, index);
                index++;
            }
        }
        roots[rootState+1].setAction(&roots[rootState], chosenAction);
        rootIndex = outcomes[rootIndex][chosenAction];
        if(roots[rootState+1].isEndState()){
            fout<<chosenAction;
            break;
        }
        else{
            fout<<chosenAction<<',';
        }
    }
    fout<<'\n';
    fout.close();
    int numStates = rootState;
    Data* game = new Data[numStates];
    double finalScore = roots[numStates-1].getScore();
    for(int i=0; i<numStates; i++){
        game[i] = Data(&roots[i], finalScore);
    }
    dq->enqueue(game, numStates);
    return finalScore;
}

int Trainer::evalGame(){ // return index of the final state in roots = numStates - 1.
    roots[0].initialize();
    rootIndex = 0;
    initializeNode(roots[0], 0);
    index = 1;
    
    int chosenAction;
    for(rootState=0; rootState<maxTime*2; rootState++){
        if(roots[rootState].actionType == 0){
            for(int j=0; j<numPaths; j++){
                expandPath();
            }
            computeActionProbs();
            chosenAction = optActionProbs();
        }
        else{
            chosenAction = getRandomChanceAction(&roots[rootState]);
            if(outcomes[rootIndex][chosenAction] == -1){
                outcomes[rootIndex][chosenAction] = index;
                Environment env;
                env.setAction(&roots[rootState], chosenAction);
                initializeNode(env, index);
                index++;
            }
        }
        roots[rootState+1].setAction(&roots[rootState], chosenAction);
        rootIndex = outcomes[rootIndex][chosenAction];
        if(roots[rootState+1].isEndState()){
            break;
        }
    }
    int numStates = rootState;
    return numStates - 1;
}

void Trainer::printGame(){
    /*
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
    }*/
}

void Trainer::exportGame(){
    roots[0].initialize();
    cout<<(roots[0].applex * boardy + roots[0].appley)<<',';
    rootIndex = 0;
    initializeNode(roots[0], 0);
    index = 1;
    
    int chosenAction;
    for(rootState=0; rootState<maxTime*2; rootState++){
        if(roots[rootState].actionType == 0){
            for(int j=0; j<numPaths; j++){
                expandPath();
            }
            computeActionProbs();
            chosenAction = optActionProbs();
        }
        else{
            chosenAction = getRandomChanceAction(&roots[rootState]);
            if(outcomes[rootIndex][chosenAction] == -1){
                outcomes[rootIndex][chosenAction] = index;
                Environment env;
                env.setAction(&roots[rootState], chosenAction);
                initializeNode(env, index);
                index++;
            }
        }
        roots[rootState+1].setAction(&roots[rootState], chosenAction);
        rootIndex = outcomes[rootIndex][chosenAction];
        if(roots[rootState+1].isEndState()){
            cout<<chosenAction;
            break;
        }
        else{
            cout<<chosenAction<<',';
        }
    }
    cout<<'\n';
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
        cout<<roots[endState].getScore()<<' ';
        scoreSum += roots[endState].getScore();
        sizeSum += roots[endState].snakeSize;
        scoreSquareSum += squ(roots[endState].getScore());
        if(roots[endState].snakeSize == boardx * boardy){
            numCompletes++;
            timeSum += roots[endState].timer;
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
    int currNode = rootIndex;
    int nextNode,nextAction;
    int count = 0;
    int currType;
    int maxIndex;
    double maxVal,candVal;
    int i;
    Environment env;
    env.copyEnv(&roots[rootState]);
    while(currNode != -1 && !env.isEndState()){
        path[count] = currNode;
        count++;
        maxVal = -1000000;
        currType = env.actionType;
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
                    candVal = sumScore[nextNode] / subtreeSize[nextNode] + 0.5 * log(subtreeSize[currNode]) / sqrt(subtreeSize[nextNode]);
                }
                if(currType == 1){
                    candVal = (double)rand() / RAND_MAX - subtreeSize[nextNode];
                }
            }
            if(candVal > maxVal){
                maxVal = candVal;
                maxIndex = i;
            }
        }
        nextAction = maxIndex;
        currNode = outcomes[currNode][maxIndex];
        env.makeAction(maxIndex);
    }
    double newVal;
    if(currNode == -1){
        outcomes[path[count-1]][nextAction] = index;
        initializeNode(env, index);
        if(hard_code){
            newVal = env.snakeSize - env.actionType;
            newVal += (maxTime - env.timer) * 0.05;
            newVal -= (abs(env.headx - env.applex)) * 0.05;
            newVal -= (abs(env.heady - env.appley)) * 0.05;
        }
        else{
            env.inputSymmetric(a.input, rand()%8);
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
        newVal = env.getScore();
        path[count] = currNode;
        count++;
    }
    for(i=0; i<count; i++){
        subtreeSize[path[i]]++;
        sumScore[path[i]] += newVal;
    }
}

void Trainer::printTree(){
    /*
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
    fout.close();*/
}

void Trainer::computeActionProbs(){
    int i;
    int nextIndex;
    for(i=0; i<numAgentActions; i++){
        nextIndex = outcomes[rootIndex][i];
        if(nextIndex != -2){
            actionProbs[i] = pow(subtreeSize[nextIndex], 2); // squ(size[nextIndex]);
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

