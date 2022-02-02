/*

Snake end value training with Monte Carlo Tree Search.
Uses MARL framework.

*/

#include "snake.h"

// states and dq are too big to be defined in the Trainer class, so they are defined outside.
Environment states[maxStates];
DataQueue dq;

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
}

int main()
{
    srand((unsigned)time(NULL));
    ofstream fout(outAddress);
    fout.close();
    ofstream netOut(netAddress);
    netOut.close();
    
    /*
    Agent a;
    a.initInput(2, 6, 6, 3, 3);
    a.addConvLayer(7, 4, 4, 3, 3);
    a.addDenseLayer(60);
    a.addDenseLayer(1);
    a.randomize();
    a.save();
    
    Environment env;
    env.initialize();
    env.applex = 1;
    env.appley = 2;
    env.timer = 2;
    env.print();
    
    env.inputSymmetric(&a.input, 0);
    a.pass();
    
    
    double output = a.output;
    double error = squ(output - 3);
    cout<<a.output<<'\n';
    a.expected = 3;
    a.backProp();
    double Dw = a.il.DparamWeights[0][0];
    a.il.paramWeights[0][0] += 0.01;
    a.pass();
    double newOutput = a.output;
    double newError = squ(newOutput - 3);
    cout<<"Calculated derivative: "<<(newError - error) / 0.01<<'\n';
    cout<<"Network derivative: "<<Dw<<'\n';
    */
    
    Trainer t(states, &dq);
    t.a.initInput(3, 6, 6, 3, 3);
    t.a.addConvLayer(4, 4, 4, 3, 3);
    t.a.addDenseLayer(30);
    t.a.addDenseLayer(1);
    t.a.randomize();
    
    for(int i=0; i<numGames; i++){
        if(i%evalPeriod == 0){
            cout<<"Game "<<i<<'\n';
            t.evaluate();
            survey(&t, i);
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



