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
    net.initInput(6, 6, 6, 3, 3);
    net.addConvLayer(9, 6, 6, 3, 3);
    net.addPoolLayer(9, 3, 3);
    net.addDenseLayer(60);
    net.addDenseLayer(1);
    net.quickSetup();
}

void trainCycle(){
    standardSetup(t.a);
    const int storeNets = 30;
    Agent nets[storeNets];
    Agent bestNet;
    double scores[storeNets];
    double bestScore;
    for(int i=0; i<storeNets; i++){
        standardSetup(nets[i]);
        scores[i] = 0;
    }
    standardSetup(bestNet);
    bestScore = 0;
    
    dq.index = 0;
    
    string netStore = "snakeConv.out";
    
    double sum = 0;
    for(int i=0; i<=numGames; i++){
        double score = t.trainTree();
        //cout<<score<<' ';
        sum += score;
        if(i>0 && i%evalPeriod == 0){
            cout<<"\nAVERAGE: "<<(sum / evalPeriod)<<'\n';
            sum = 0;
        }
        scores[i%storeNets] = score;
        t.a.save(netStore);
        nets[i%storeNets].readNet(netStore);
        double scoreSum = 0;
        for(int j=0; j<storeNets; j++){
            scoreSum += scores[j];
        }
        if(scoreSum > bestScore){
            bestScore = scoreSum;
            cout<<"Best score: "<<(bestScore / storeNets)<<'\n';
            int maxIndex = 0;
            for(int j=1; j<storeNets; j++){
                if(scores[maxIndex] < scores[j]){
                    maxIndex = j;
                }
            }
            cout<<"Max score: "<<scores[maxIndex]<<'\n';
            nets[maxIndex].save(netStore);
            bestNet.readNet(netStore);
        }
    }
    
    for(int i=0; i<10; i++){
        ofstream fout(outAddress, ios::app);
        fout<<"Printed game "<<i<<'\n';
        fout.close();
        t.printGame();
    }
    
    bestNet.save("snakeConv.out");
}

void evaluate(){
    standardSetup(t.a);
    t.a.readNet("snakeConv.out");
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

int main()
{
    srand((unsigned)time(NULL));
    start_time = time(NULL);
    /*
    ofstream fout1("small.out");
    fout1.close();
    ofstream fout2("med.out");
    fout2.close();
    ofstream fout3("large.out");
    fout3.close();
    ofstream fout5("SnakeConvResults.out");
    fout5.close();
    ofstream fout4(outAddress);
    fout4.close();
    t.a.netIn = new ifstream("SnakeConvIn.txt");
    t.a.netOut = new ofstream("SnakeConvOut.txt");
    t.a.initInput(4, 4, 4, 3, 3);
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
     */
    
    //trainCycle();
    
    evaluate();
    
    return 0;
    
}



