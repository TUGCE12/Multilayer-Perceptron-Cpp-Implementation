/* For this project:
  * StatQuest with Josh Starmer / Neural Networks / Deep Learning /-
  * -> https://www.youtube.com/watch?v=zxagGtF9MeU&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=2
  * The first 8 videos in the playlist were used.
  * There is no code in the video series (at least in the first 8)
  * It is explained theoretically only.
  * This project is a C++ implementation of the theoretically explained MLP problem and BackPropagation example.
  * Example Problem: A pharmaceutical company has tested 3 groups for the drug it produces.
  * 1. Group: Low dose drug was applied. Viruses are not dead. (Unsuccessful)
  * 2. Group: Normal dose of drug was applied. The viruses are dead. (Successful)
  * 3rd Group: High dose drug was applied. Viruses are not dead. (Unsuccessful)
  * For simplicity, drug doses are taken as low = 0 and high = 1.
  * Success =1 and Fail = 0 .
  * Dataset = [[0.0,0.0],[0.5,1.0],[1.0,0.0]] .
  * Neural Network will be used to find a model that will fit this.
  |     *
  |
  |*__________*__ The data distribution is like this.
  * To draw a graph suitable for this data, I need 2 graphs in the form of y = ax + b.
  * In video was used 2 nodes in Hidden Layer.
  *
  * *****************************************
Number of input nodes = 1
Number of nodes per hidden layer = 2
Number of output nodes = 1
Number of hidden layers = 1
The learning rate = 0.015
Loss Fuction = Sum of squares Residual
Activation function = SoftPlus = f(x) = log(1+e^x)



Redidual = Observed - Predicted


 Neural Network:
n= Number of nodes per hidden layer
i = Number of data
__________                                    __________________                                ________
|Input[i]|---(x W1[n] )---( + B1[n] )---------|Hidden Layer (n)|---( x W2[n] )------(Sum) + B---|Output|
|________|                                    |________________|                                |______|

 (-->SumFunction +B for a x)
 Neural Network Function is = E(i=0->i=2) [B + W2i * softPlus( x * W1i + B1i )]

 E = Sum
 * */

#include <iostream>
#include<vector>
using namespace std;
#include <cmath>
#define N 30 //Number of nodes per hidden layer = 2 ---> updated to 30 because 30 nodes performs better than 2
#define Max_Iter 14000
#define zero 0.000000001
#define Precision 0.00000001
#define reductionLearninRate 0.99988

double B = 0;
double W1[N] = {};
double B1[N] = {};
double W2[N] = {};
double LR = 0.015; //Learning Rate
int flag_W1 = 1;
int flag_W2 = 1;
int flag_B1 = 1;
int flag_B = 1;



//if return nan we use iteration back
double oldB = 0;
double oldW1[N] = {};
double oldB1[N] = {};
double oldW2[N] = {};



//set all Bias to 0
void set_Weight_Bias(){
    srand(time(NULL));
    for(int i = 0; i<N; i++){
        B1[i]=0;
        W1[i] = 2 * rand() / RAND_MAX -1;
        W2[i] = 2 * rand() / RAND_MAX -1;

    }
}

double round_to(double value, double precision = 1.0)
{
    return std::round(value / precision) * precision;
}

// Activation Function: Define Function for calculate SoftPlus=log(1+e^x)
double softPlus(double x) {
    double num = log10(1+exp(x));
    return round_to(num, Precision);
}

double SumFunction_PlusB(double x){ // B+ W2i * softPlus( x * W1i + B1i )
    double sum = B;
    for(int i = 0 ; i < N; i++){
        sum += W2[i] * softPlus( x * W1[i] + B1[i]);
    }
    return round_to(sum,Precision);
}

//Get the prediction for x to yp (predicted y):

vector <pair<double,double>> Predicted (vector <pair<double,double>> data){
    vector <pair<double,double>> prediction = {};
    for (int i = 0; i< data.size();i++){
        prediction.push_back(make_pair(data[i].first,SumFunction_PlusB(data[i].first)));
    }
    return prediction;
}


//print dataset for prediction
void PrintPredicted (vector <pair<double,double>> data){
    vector <pair<double,double>> prediction = {};
    for (int i = 0; i< data.size();i++){
        prediction.push_back(make_pair(data[i].first,SumFunction_PlusB(data[i].first)));
        cout <<"x= "<< prediction[i].first<<", yp= "<<prediction[i].second<<endl;
    }
}
//e^x/(1+e^x) calculation
double expCalcul (double x){
    return round_to( exp(x) / ( 1 + exp(x) ), Precision );
}

//optimizing step the curve
//For Gradiant descent: **** Step Size = Derivative x Learning Rate and *** New Parameter = Old Parameter - Step Size
int GradientDescent (vector <pair<double,double>> data){
    vector <pair<double,double>> predictions = Predicted(data);
    //for W1 update:
    double devW1 = 0;
    double devW2 = 0;
    double devB1 = 0;
    double devB = 0;

    for (int n = 0; n< N; n++){
        oldW1[n] = W1[n];
        oldB1[n] = B1[n];
        oldW2[n] = W2[n];
    }
    oldB = B;

    for (int n = 0; n< N; n++){ //n = Number of nodes per hidden layer
        for (int i = 0; i< data.size();i++){  //i = number of data
            //Derivative
            double ssrDerivP = -2 * (data[i].second - predictions[i].second);
            devW1 += ssrDerivP * W2[n] * expCalcul(data[i].first) * data[i].first;
            devW2 += ssrDerivP * softPlus(W1[n] * data[i].first + B1[n]);
            devB1 += ssrDerivP * W2[i] * expCalcul(data[i].first);
            if(n == 0){
                devB += ssrDerivP;
            }
        }
        //New Parameter for weights and biases
        if (flag_W1 == 1 ){
            W1[n] -= devW1 * LR;
            if (::abs(devW1 * LR) <= zero){
                flag_W1 = 0;
            }
        }
        if (flag_W2 == 1 ){
            W2[n] -= devW2 * LR;
            if (::abs(devW2 * LR) <= zero){
                flag_W2 = 0;
            }
        }
        if (flag_B1 == 1 ){
            B1[n] -= devB1 * LR;
            if (::abs(devB1 * LR) <= zero){
                flag_B1 = 0;
            }
        }
    }

    if (flag_B == 1 ){
        B -= devB * LR;
        if (::abs(devB * LR) <= zero){
            flag_B = 0;
        }
    }

    //new value check if nan return old value
    vector <pair<double,double>> predictions2 = Predicted(data);
    for (int i = 0; i< data.size();i++){
        if (::isnan(predictions2[i].second)){
            for (int n = 0; n< N; n++){
                W1[n] = oldW1[n];
                B1[n] = oldB1[n];
                W2[n] = oldW2[n];
            }
            B = oldB;
            return -1;
        }
    }
    LR = LR * reductionLearninRate;
    return 1;

}


void Train(vector <pair<double,double>> dataSet){
    std::cout<< "\n\nBefore the train :\n"<< endl;
    PrintPredicted(dataSet);
    int control = 0;

    for(int i = 0 ; i < Max_Iter; i++){

        if (flag_W1 == 0 and  flag_W2 == 0 and  flag_B1== 0 and flag_B==0){ // from stepsize = Derivation of SSR
            cout<<"\n\nAfter "<<i<<"th iteration step size close to 0"<<endl;
            cout<<"End of the train..."<<endl;
            PrintPredicted(dataSet);
            return;
        }else{
            flag_W1 = 1;
            flag_W2 = 1;
            flag_B1 = 1;
            flag_B = 1;
        }

        control = GradientDescent(dataSet);

        if(control == -1){
            cout<<"\n\nAfter "<<i-1<<"th iteration prediction(s) value going to be nan"<<endl;
            cout<<"End of the train..."<<endl;
            PrintPredicted(dataSet);
            return;

        }

        if(i == Max_Iter/2 or i == Max_Iter/4  or i == Max_Iter*3/4 or i == Max_Iter*9/16){
            std::cout<< "\n"<< endl;
            std::cout<<i<< " th of training\n :"<< endl;
            PrintPredicted(dataSet);
        }

    }
    std::cout << "\n\nEnd of train :\n"<< endl;
    PrintPredicted(dataSet);

}

int main() {

    vector <pair<double,double>> dataSet ={{0.0,0.0},
                                           {0.5,1.0},
                                           {1.0,0.0}};


    set_Weight_Bias();
    /*
    std::cout<<"First prediction\n"<< endl;
    PrintPredicted(dataSet);
    GradientDescent(dataSet);
    std::cout<<"\nafter Gradient descent second Prediction\n"<< endl;
    PrintPredicted(dataSet);

    GradientDescent(dataSet);
    std::cout<<"\nafter Gradient descent 3 Prediction\n"<< endl;
    PrintPredicted(dataSet);
     */

    double x = 0.0014567;
    double p = SumFunction_PlusB(x);
    cout<<"Before Train predicted value of "<<x<<" equal to : "<<p<<endl;
    Train(dataSet);
    p = SumFunction_PlusB(x);
    cout<<"\n**********\nAfter Train predicted value of "<<x<<" equal to : "<<p<<endl;

    return 0;
}
