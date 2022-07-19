/*

Snake end value training with Monte Carlo Tree Search.
Uses MARL framework.

*/

#include "snake.h"

// dq is too big to be defined in the Trainer class, so it is defined outside.
DataQueue dq;
Trainer t(&dq);

unsigned long start_time;

void standardSetup(Agent& net){
    net.commonBranch.initEnvironmentInput(10, 10, 10, 3, 3);
    net.commonBranch.addConvLayer(10, 10, 10, 3, 3);
    net.commonBranch.addPoolLayer(10, 5, 5);
    net.setupCommonBranch();
    net.policyBranch.addFullyConnectedLayer(200);
    net.policyBranch.addFullyConnectedLayer(100);
    net.policyBranch.addOutputLayer(4);
    net.valueBranch.addFullyConnectedLayer(200);
    net.valueBranch.addFullyConnectedLayer(100);
    net.valueBranch.addOutputLayer(1);
    net.setup();
    net.randomize(0.2);
}

void printArray(double* A, int size){
    for(int i=0; i<size; i++){
        cout<<A[i]<<' ';
    }
    cout<<'\n';
}

void testNet(){
    cout<<"Testing net:\n";
    Agent net;
    standardSetup(net);
    net.randomize(0.5);
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            net.input->snake[i][j] = rand() % 5 - 1;
        }
    }
    for(int i=0; i<3; i++){
        for(int j=0; j<2; j++){
            net.input->pos[i][j] = rand() % boardx;
        }
        //net.input->param[i] = (double) rand() / RAND_MAX;
    }
    for(int i=0; i<4; i++){
        net.validAction[i] = rand() % 2;
    }
    net.valueExpected = (double) rand() / RAND_MAX;
    double sum = 0;
    for(int i=0; i<4; i++){
        if(net.validAction[i]){
            net.policyExpected[i] = (double) rand() / RAND_MAX;
            sum += net.policyExpected[i];
        }
    }
    double sum2 = 0;
    for(int i=0; i<4; i++){
        if(net.validAction[i]){
            net.policyExpected[i] /= sum;
            sum2 += net.policyExpected[i];
        }
    }
    
    net.pass(PASS_FULL);
    
    double base = squ(net.valueOutput - net.valueExpected);
    for(int i=0; i<4; i++){
        if(net.validAction[i]){
            base -= net.policyExpected[i] * log(net.policyOutput[i]);
        }
    }
    net.backProp(PASS_FULL);
    double ep = 0.000001;
    
    for(int l=0; l<net.numLayers; l++){
        for(int i=0; i<net.layers[l]->numParams; i++){
            net.layers[l]->params[i] += ep;
            net.pass(PASS_FULL);
            net.layers[l]->params[i] -= ep;
            double new_error = squ(net.valueOutput - net.valueExpected);
            for(int j=0; j<4; j++){
                if(net.validAction[j]){
                    new_error -= net.policyExpected[j] * log(net.policyOutput[j]);
                }
            }
            //cout<< ((new_error - base) / ep) << ' ' << net.layers[l]->Dparams[i]<<'\n';
            assert( abs((new_error - base) / ep - net.layers[l]->Dparams[i]) < 0.0001);
        }
    }
}

void trainCycle(){
    cout<<"Beginning training: "<<time(NULL)<<'\n';
    standardSetup(t.a);

    //cout<<"Reading net:\n";
    //t.a.readNet("snakeConv.in");

    const int storePeriod = 50;
    
    dq.index = 0;
    dq.momentum = 0.7;
    dq.learnRate = 0.01;
    t.actionTemperature = 2;
    
    //cout<<"Reading games\n";
    //vector<int> scores = dq.readGames(); // read games from games.in file.
    //cout<<"Finished reading " << dq.index << " games\n";
    vector<int> scores;
    
    double sum = 0;
    int completions = 0;
    double completionTime = 0;
    
    string gameLog = "gameLog.out";
    string summaryLog = "summary.out";
    string valueLog = "valueLog.out";
    string scoreLog = "scores.out";
    ofstream hold(gameLog);
    hold.close();
    ofstream hold2(summaryLog);
    hold2.close();
    ofstream hold3(valueLog);
    hold3.close();
    ofstream hold4(scoreLog);
    hold4.close();
    t.gameLog = gameLog;
    t.valueLog = valueLog;
    
    for(int i=0; i<=numGames; i++){
        ofstream fout(gameLog, ios::app);
        fout<<"Game "<<i<<' '<<time(NULL)<<'\n';
        fout.close();
        ofstream valueOut(valueLog, ios::app);
        valueOut<<"Game "<<i<<' '<<time(NULL)<<'\n';
        valueOut.close();

        double result = t.trainTree();
        double score = min(result, boardx*boardy);
        sum += score;
        if(score == boardx*boardy){
            completions++;
            completionTime += result - score;
        }

        cout<<i<<':'<<score<<' ';
        
        ofstream summaryOut(summaryLog, ios::app);
        summaryOut<<i<<':'<<score<<' ';
        summaryOut.close();

        scores.push_back(score);
        ofstream scoreOut(scoreLog);
        for(int s=0; s<scores.size(); s++){
            if(s > 0){
                scoreOut<<',';
            }
            scoreOut<<scores[s];
        }
        scoreOut<<'\n';
        int AVG_PERIOD = 10;
        for(int s=0; s<scores.size() / AVG_PERIOD; s++){
            if(s > 0){
                scoreOut<<',';
            }
            double sum = 0;
            for(int j=0; j<AVG_PERIOD; j++){
                sum += scores[s * AVG_PERIOD + j];
            }
            scoreOut<<sum;
        }
        scoreOut.close();
        /*
        maxScore = max(maxScore, score);
        
        if(maxScore >= 10){
            //t.actionTemperature = max(t.actionTemperature, 2);
        }
        if(maxScore >= 40){
            //dq.learnRate = 0.0003 / (1 + maxScore);
            //t.actionTemperature = max(t.actionTemperature, 3);
        }
        if(maxScore >= 100){
            //dq.learnRate = 0.00015 / (1 + maxScore);
        }
        */
        if(i>0 && i%evalPeriod == 0){
            cout<<"\nAVERAGE: "<<(sum / evalPeriod)<<" in iteration "<<i<<'\n';
            cout<<"Completions: "<<((double) completions / evalPeriod)<<'\n';
            if(completions > 0){
                cout<<"Average completion time: "<<(completionTime / completions)<<'\n';
            }
            cout<<" TIMESTAMP: "<<(time(NULL) - start_time)<<'\n';
            sum = 0;
            completions = 0;
            completionTime = 0;
        }
        if(i % storePeriod == 0){
            t.a.save("nets/Game" + to_string(i) + ".out");
        }
        
        dq.trainAgent(t.a);
    }
}

void evaluate(){
    standardSetup(t.a);
    t.a.readNet("snakeConv.in");
    t.evaluate();
    
    ofstream fout4(outAddress);
    fout4.close();
    for(int i=0; i<1; i++){
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

int main()
{
    srand((unsigned)time(NULL));
    start_time = time(NULL);
    
    /*
    for(int i=0; i<10; i++){
        testNet();
    }*/
    
    trainCycle();
    
    //evaluate();
    
    //manual_game();
    
    //exportGames();

    //runDeterministic();
    
    //checkDeterministic();
    
    //dq.readGames();
    
    /*
    Environment env;
    env.initialize();
    env.log();
    standardSetup(t.a);
    env.inputSymmetric(t.a, 0);
    
    t.a.pass(PASS_FULL);
    cout<<t.a.valueOutput<<'\n';
    for(int i=0; i<4; i++){
        cout<<t.a.policyOutput[i]<<' ';
    }
    cout<<'\n';
    t.a.save("neato.out");
    for(int i=0; i<t.a.numLayers; i++){
        cout<<t.a.layers[i]->inputs<<' '<<t.a.layers[i]->Dinputs<<' '<<t.a.layers[i]->outputs<<' '<<t.a.layers[i]->Doutputs<<'\n';
    }
    cout<<'\n';
    */
    
    return 0;
    
}



