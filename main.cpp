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
    net.addDenseLayer(60);
    net.addDenseLayer(1);
    net.quickSetup();
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
    net.expected = (double) rand() / RAND_MAX;
    
    net.pass();
    double base = squ(net.output - net.expected);
    net.backProp();
    double ep = 0.000001;
    
    for(int l=0; l<net.numLayers; l++){
        for(int i=0; i<net.layers[l]->numParams; i++){
            net.layers[l]->params[i] += ep;
            net.pass();
            net.layers[l]->params[i] -= ep;
            double new_error = squ(net.output - net.expected);
            //cout<< ((new_error - base) / ep) << ' ' << net.layers[l]->Dparams[i]<<'\n';
            assert( abs((new_error - base) / ep - net.layers[l]->Dparams[i]) < 0.001);
        }
    }
}

void trainCycle(){
    standardSetup(t.a);
    const int storePeriod = 50;
    
    dq.index = 0;
    dq.learnRate = 0.03;
    dq.momentum = 0.9;
    
    t.a.maxValue = 0;
    
    double sum = 0;
    int completions = 0;
    for(int i=0; i<=numGames; i++){
        double score = t.trainTree();
        cout<<score<<' ';
        
        if(score >= 40){
            dq.learnRate = min(dq.learnRate, 0.01);
        }
        
        t.a.maxValue = max(t.a.maxValue, score);
        
        sum += score;
        if(score >= 40){
            completions++;
        }
        if(i>0 && i%evalPeriod == 0){
            cout<<"\nAVERAGE: "<<(sum / evalPeriod)<<" in iteration "<<i<<'\n';
            cout<<"Completions: "<<((double) completions / evalPeriod)<<'\n';
            cout<<" TIMESTAMP: "<<(time(NULL) - start_time)<<'\n';
            sum = 0;
            completions = 0;
        }
        if(i % storePeriod == 0){
            t.a.save("nets/Game" + to_string(i) + ".out");
        }
        
        dq.trainAgent(&t.a);
    }
}

void evaluate(){
    standardSetup(t.a);
    t.a.readNet("snakeConv.in");
    t.evaluate();
    
    ofstream fout4(outAddress);
    fout4.close();
    for(int i=0; i<10; i++){
        ofstream fout(outAddress, ios::app);
        fout<<"Printed game "<<i<<'\n';
        fout.close();
        t.printGame();
    }
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



