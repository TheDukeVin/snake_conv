/*

Snake end value training with Monte Carlo Tree Search.
Uses MARL framework.

*/

#include "snake.h"

// states and dq are too big to be defined in the Trainer class, so they are defined outside.
Environment states[maxStates];
DataQueue dq;
/*
void survey(Trainer* t, int nGames){
    ofstream fout(outAddress, ios::app);
    fout<<"Survey "<<nGames<<'\n';
    fout.close();
    int N = min(dq.index, queueSize);
    for(int i=0; i<N; i++){
        Data* data = dq.queue[i];
        data->e.print();
        ofstream fout(outAddress, ios::app);
        fout<<"Expected: "<<data->expectedValue<<'\n';
        data->e.inputSymmetric(&t->a.input, rand() % 8);
        t->a.pass();
        fout<<"Network: "<<t->a.output<<'\n';
        fout.close();
    }
}*/

int main()
{
    srand((unsigned)time(NULL));
    ofstream fout(outAddress);
    fout.close();
    ofstream netOut(netAddress);
    netOut.close();
    
    Trainer t(states, &dq);
    t.a.initInput(3, 6, 6, 3, 3);
    t.a.addConvLayer(4, 4, 4, 3, 3);
    t.a.addDenseLayer(30);
    t.a.addDenseLayer(1);
    t.a.randomize();
    t.a.resetGradient();
    
    
    dq.batchSize = 500;
    dq.queueSize = 2000;
    dq.mult = 0.05 / dq.batchSize;
    
    int goalsReached = 0;
    
    for(int i=0; i<numGames; i++){
        if(i%evalPeriod == 0){
            cout<<"Game "<<i<<'\n';
            t.evaluate();
            //survey(&t, i);
        }
        double score = t.trainTree();
        //cout<<score<<'\n';
        /*
        if(score > 4 && goalsReached < 1){
            dq.batchSize = 900;
            dq.queueSize = 4000;
            dq.mult = 0.02 / dq.batchSize;
            cout<<"Passed score 4\n";
            goalsReached = 1;
        }
        if(score > 8 && goalsReached < 2){
            dq.batchSize = 2000;
            dq.queueSize = 15000;
            dq.mult = 0.01 / dq.batchSize;
            cout<<"Passed score 8\n";
            goalsReached = 2;
        }*/
    }
    for(int i=0; i<10; i++){
        ofstream fout(outAddress, ios::app);
        fout<<"Printed game "<<i<<'\n';
        fout.close();
        t.printGame();
    }
    
    
    return 0;
    
}



