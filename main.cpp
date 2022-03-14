/*

Snake end value training with Monte Carlo Tree Search.
Uses MARL framework.
 
Experiment 1:
testing out impact of starting parameter range.

*/

#include "snake.h"

// states and dq are too big to be defined in the Trainer class, so they are defined outside.
Environment states[maxStates];
DataQueue dq;
Trainer t(states, &dq);

unsigned long start_time;

void run_trial(int size){
    ofstream* fout;
    if(size == 0) fout = new ofstream("small.out", ios::app);
    if(size == 1) fout = new ofstream("med.out", ios::app);
    if(size == 2) fout = new ofstream("large.out", ios::app);
    if(size == 0) t.a.randomize(0.05);
    if(size == 1) t.a.randomize(0.2);
    if(size == 2) t.a.randomize(0.5);
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
}

int main()
{
    srand((unsigned)time(NULL));
    start_time = time(NULL);
    ofstream fout1("small.out");
    fout1.close();
    ofstream fout2("med.out");
    fout2.close();
    ofstream fout3("large.out");
    fout3.close();
    ofstream fout4(outAddress);
    fout4.close();
    t.a.netIn = new ifstream("SnakeConvIn.txt");
    t.a.netOut = new ofstream("SnakeConvOut.txt");
    t.a.initInput(3, 6, 6, 3, 3);
    t.a.addConvLayer(4, 4, 4, 3, 3);
    t.a.addDenseLayer(30);
    t.a.addDenseLayer(1);
    t.a.setupIO();
    t.a.randomize(0.2);
    t.a.resetGradient();
    
    for(int i=0; i<40; i++){
        run_trial(0);
        run_trial(1);
        run_trial(2);
    }
    /*
    t.a.randomize(0.2);
    
    for(int i=0; i<10; i++){
        ofstream fout(outAddress, ios::app);
        fout<<"Printed game "<<i<<'\n';
        fout.close();
        t.printGame();
    }
    t.a.save();
     */
    
    t.a.netIn->close();
    t.a.netOut->close();
    return 0;
    
}



