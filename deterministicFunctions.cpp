
#include "snake.h"

void Environment::getDeterministicFeatures(double* features){
    features[0] = (double) timer / 100;
    features[1] = (double) score;
    features[2] = (double) actionType;
    features[3] = (double) (abs(headx - applex) + abs(heady - appley)) / boardx;

    int dist[boardx * boardy];
    for(int i=0; i<boardx * boardy; i++){
        dist[i] = -1;
    }
    list<int> queue;
    queue.push_back(headx * boardy + heady);
    dist[headx * boardy + heady] = 0;
    int numVisibleNodes = 0;
    while(!queue.empty()){
        int currNode = queue.front();
        queue.pop_front();
        numVisibleNodes++;
        int currx = currNode / boardy;
        int curry = currNode % boardy;
        for(int i=0; i<4; i++){
            int nextx = currx + dir[i][0];
            int nexty = curry + dir[i][1];
            if(nextx != -1 && nextx != boardx && nexty != -1 && nexty != boardy && snake[nextx][nexty] == -1 && dist[nextx * boardy + nexty] == -1){
                dist[nextx * boardy + nexty] = dist[currNode] + 1;
                queue.push_back(nextx * boardy + nexty);
            }
        }
    }

    features[4] = (double) dist[applex * boardy + appley] / boardx;
    features[5] = (double) numVisibleNodes / (boardx * boardy);

    int numOneNeighbor = 0;
    int numZeroNeighbor = 0;
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            if(snake[i][j] != -1){
                continue;
            }
            int numNeighbor = 0;
            for(int d=0; d<4; d++){
                int nextx = i + dir[d][0];
                int nexty = j + dir[d][1];
                if(nextx != -1 && nextx != boardx && nexty != -1 && nexty != boardy && snake[nextx][nexty] == -1){
                    numNeighbor++;
                }
            }
            if(numNeighbor == 1){
                numOneNeighbor++;
            }
            if(numNeighbor == 0){
                numZeroNeighbor++;
            }
        }
    }

    features[6] = (double) numOneNeighbor;
    features[7] = (double) numZeroNeighbor;
}


double LinearModel::pass(double* features){
    double sum = 0;
    for(int i=0; i<numFeatures; i++){
        sum += features[i] * weights[i];
    }
    return sum + bias[0];
}

void LinearModel::backProp(double* features, double expected){
    double currValue = pass(features);
    //cout<<"DIFFERENCE: "<<(currValue - expected)<<'\n';
    for(int i=0; i<numFeatures; i++){
        Dparams[i] += 2 * (currValue - expected) * features[i];
    }
    Dbias[0] += 2 * (currValue - expected);
}

void LinearModel::resetGradient(){
    for(int i=0; i<numParams; i++){
        Dparams[i] = 0;
    }
}

void LinearModel::updateParameters(double mult, double momentum){
    for(int i=0; i<numParams; i++){
        //cout<<Dparams[i]<<'\n';
        params[i] -= Dparams[i] * mult;
        Dparams[i] *= momentum;
    }
}
