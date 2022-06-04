/*

Snake end value training with Monte Carlo Tree Search.
Uses MARL framework.

*/

#include "snake.h"

// states and dq are too big to be defined in the Trainer class, so they are defined outside.
Environment states[maxStates];
DataQueue dq;
Trainer t(states, &dq);

unsigned long start_time;

/*
void run_trial(int size){
    ofstream* fout = new ofstream("SnakeConvResults.out", ios::app);
    t.a.randomize(0.3);
    t.a.resetGradient();
    dq.index = 0;
    
    for(int i=0; i<numGames; i++){
        if(i%evalPeriod == 0){
            cout<<"Game "<<i<<'\n';
            (*fout)<<t.evaluate()<<", ";
        }
        t.trainTree();
    }
    (*fout)<<"TIME: "<<(time(NULL) - start_time)<<'\n';
    fout->close();
}*/


void standardSetup(Agent& net){
    net.initInput(6, 6, 6, 5, 5);
    net.addConvLayer(9, 6, 6, 5, 5);
    net.addPoolLayer(9, 3, 3);
    net.addDenseLayer(80);
    net.addOutputLayer(5);
    net.randomize(0.2);
}

double error(double* A, double* B, int size){
    double sum = 0;
    for(int i=0; i<size; i++){
        sum += squ(A[i] - B[i]);
    }
    return sum;
}

void testNet(){
    Agent net;
    standardSetup(net);
    net.randomize(0.3);
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            net.input->snake[i][j] = rand() % 5 - 1;
        }
    }
    for(int i=0; i<3; i++){
        for(int j=0; j<2; j++){
            net.input->pos[i][j] = rand() % 6;
        }
        net.input->param[i] = (double) rand() / RAND_MAX;
    }
    for(int i=0; i<5; i++){
        net.expected[i] = (double) rand() / RAND_MAX;
    }
    
    net.pass();
    double base = error(net.expected, net.output, 5);
    net.backProp();
    double ep = 0.00001;
    
    for(int l=0; l<net.numLayers; l++){
        for(int i=0; i<net.layers[l]->numParams; i++){
            net.layers[l]->params[i] += ep;
            net.pass();
            net.layers[l]->params[i] -= ep;
            double new_error = error(net.expected, net.output, 5);
            cout<<l<<' '<< ((new_error - base) / ep) << ' ' << net.layers[l]->Dparams[i]<<'\n';
            assert( abs((new_error - base) / ep - net.layers[l]->Dparams[i]) < 0.001);
        }
    }
}

void trainCycle(){
    standardSetup(t.a);
    const int storePeriod = 50;
    
    dq.index = 0;
    dq.learnRate = 0.02;
    dq.momentum = 0.9;
    
    double sum = 0;
    for(int i=0; i<=numGames; i++){
        double score = t.trainTree();
        //cout<<score<<' ';
        cout<<"SCORE: "<<score<<'\n';
        
        if(score >= 10){
            dq.learnRate = min(dq.learnRate,0.001);
        }
        if(score >= 20){
            dq.learnRate = min(dq.learnRate,0.0002);
        }
        if(score >= 30){
            dq.learnRate = min(dq.learnRate,0.0001);
        }
        
        sum += score;
        if(i>0 && i%evalPeriod == 0){
            cout<<"\nAVERAGE: "<<(sum / evalPeriod)<<" in iteration "<<i<<" TIMESTAMP: "<<(time(NULL) - start_time)<<'\n';
            sum = 0;
        }
        if(i % storePeriod == 0){
            t.a.save("nets/Game" + to_string(i) + ".out");
        }
        //cout<<"BEFORE TRAIN\n";
        dq.trainAgent(&t.a);
        
        
        double paramSum = 0;
        for(int l=0; l<t.a.numLayers; l++){
            for(int j=0; j<t.a.layers[l]->numParams; j++){
                paramSum += squ(t.a.layers[l]->params[j]);
            }
        }
        cout<<"PARAMS: "<<paramSum<<'\n';
        
        
        //cout<<"AFTER TRAIN\n";
    }
}

void evaluate(){
    standardSetup(t.a);
    t.a.readNet("snakeConv.in");
    t.printGame();
    //t.evaluate();
    /*
    ofstream fout4(outAddress);
    fout4.close();
    for(int i=0; i<10; i++){
        ofstream fout(outAddress, ios::app);
        fout<<"Printed game "<<i<<'\n';
        fout.close();
        t.printGame();
    }*/
}

void exportGames(){
    standardSetup(t.a);
    t.a.readNet("snakeConv.in");
    t.exportGame();
}

void manual_game(){
    Environment env, hold;
    env.initialize();
    char dirs[4] = {'d', 's', 'a', 'w'};
    while(!env.isEndState()){
        env.log();
        int actionIndex = -1;
        if(env.actionType == 0){
            char dir;
            cin>>dir;
            for(int i=0; i<4; i++){
                if(dir == dirs[i]){
                    actionIndex = i;
                }
            }
        }
        else{
            actionIndex = t.getRandomChanceAction(&env);
        }
        assert(actionIndex >= 0);
        hold.setAction(&env, actionIndex);
        env.copyEnv(&hold);
    }
    cout<<"Final Score: "<<env.getScore()<<'\n';
}

int main()
{
    srand((unsigned)time(NULL));
    start_time = time(NULL);
    
    //testNet();
    
    trainCycle();
    
    //evaluate();
    
    //manual_game();
    
    //exportGames();
    
    return 0;
    
}



