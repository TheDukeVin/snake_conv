/*

Snake end value training with Monte Carlo Tree Search.
Uses MARL framework.

*/

#include "snake.h"

// states and dq are too big to be defined in the Trainer class, so they are defined outside.
Environment states[maxStates];
DataQueue dq;

int main()
{
    srand((unsigned)time(NULL));
    ofstream fout(outAddress);
    fout.close();
    ofstream netOut(netAddress);
    netOut.close();
    
    Trainer t(states, &dq);
    t.a.initInput(4, 6, 6, 3, 3);
    t.a.addConvLayer(7, 4, 4, 3, 3);
    t.a.addDenseLayer(60);
    t.a.addDenseLayer(1);
    
    for(int i=0; i<numGames; i++){
        if(i%evalPeriod == 0){
            cout<<"Game "<<i<<'\n';
            t.evaluate();
        }
        t.trainTree();
    }
    for(int i=0; i<10; i++){
        ofstream fout(outAddress, ios::app);
        fout<<"Printed game "<<i<<'\n';
        fout.close();
        t.printGame();
    }
    
    t.a.save();
    
    return 0;
    
}



