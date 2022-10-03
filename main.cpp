#include <algorithm>
#include <cmath>
#include <fstream>
#include "GSA.h"
#include <iostream>
#include <map>
#include <omp.h>
#include "omprng.h"
#include <string>
#include <vector>

using namespace std;

const int NCh = 50,
RyR_size = 29;

int N_ch_exp,
N_SbS_exp,
N_Both_exp,
N_Iso_exp;

const double a = 3,
b = 19;

struct channel{
    double x;
    double y;
    bool isopen;
    channel(double a, double b) : x(a), y(b),isopen(false) { }
};


double f(double *z,int N){
    int N_ch = 0;
    int N_SbS = 0;
    int N_Both = 0;
    int N_Iso = 0;
    double penalty = 0;
    for(int i = 0; i < NCh; ++i){//#finding the type of each channel(SbS,Ch etc.)
        bool is_ch = false;
        bool is_SbS = false;
        double x1 = z[2*i];
        double y1 = z[2*i+1];
        for(int j=0; j < NCh; ++j){
            if(i==j)
                continue;
            double x2 = z[2*j];
            double y2 = z[2*j+1];
            //too close
            if(abs(x1-x2) < RyR_size && abs(y1-y2) < RyR_size){
                penalty+=1e1;
                continue;
            }
            if(abs(x1-x2) < RyR_size + a){
                if(-abs(y1-y2) + RyR_size < 0) //isolated
                    continue;
                if(-abs(y1-y2) + RyR_size <= b) //checkerboard
                    is_ch = true;
                else
                    is_SbS = true;
            }
            else{
                if(abs(y1-y2) < RyR_size + a){
                    if(-abs(x1-x2) + RyR_size < 0) //isolated
                        continue;
                    if(-abs(x1-x2) + RyR_size <= b) //checkerboard
                        is_ch = true;
                    else
                        is_SbS = true;
                }
            }
        }
        if(is_ch && is_SbS)
            N_Both+=1;
        else
            if(is_ch)
                N_ch+=1;
            else
                if(is_SbS)
                    N_SbS+=1;
                else
                    N_Iso+=1;
    }
    double N_ch_diff = N_ch_exp == 0 ? (N_ch_exp-N_ch) : (N_ch_exp-N_ch)/sqrt(N_ch_exp);
    double N_SbS_diff = N_SbS_exp == 0 ? (N_SbS - N_SbS_exp) : (N_SbS - N_SbS_exp)/sqrt(N_SbS_exp);
    double N_Both_diff = N_Both_exp == 0 ? (N_Both - N_Both_exp) : (N_Both - N_Both_exp)/sqrt(N_Both_exp);
    double N_Iso_diff = N_Iso_exp == 0 ? (N_Iso - N_Iso_exp) : (N_Iso - N_Iso_exp)/sqrt(N_Iso_exp);
    return N_ch_diff*N_ch_diff + N_SbS_diff*N_SbS_diff + N_Both_diff*N_Both_diff + N_Iso_diff*N_Iso_diff + penalty;
}

int main(int argc, char* argv[]){
    int Nthreads = 8;
    ifstream in("input_dist.txt");
    map<string,double> pars;
    string t1;
    double t2;
    while(in >> t1 >> t2 )
        pars[t1] = t2;
    N_ch_exp = pars["N_ch_exp"];
    N_SbS_exp = pars["N_SbS_exp"];
    N_Both_exp = pars["N_Both_exp"];
    N_Iso_exp = pars["N_Iso_exp"];
    in.close();
    in.open("params.txt");
    while(in >> t1 >> t2 )
        pars[t1] = t2;
    in.close();
    double mu = pars["mu"];
    double Far = 96485.33;
    cout << pars["kp"] << pars["I_RyR"] << pars["d_c"] << endl;
    double alpha = pars["kp"] * pow(pars["I_RyR"]/(2*2*acos(-1)*Far*pars["d_c"]) * 1e12,mu);
    cout << alpha << endl;
    double delta = pars["delta"];
    unsigned int Nscenarios = 10000;
    omprng MyRng;
    double *params = new double[8];
    params[0] = 400; //N
    params[1] = 3000; //tmax
    params[2] = 1000; //To
    params[3] = 0.9; //pmo
    params[4] = 0.987; //alpha
    params[5] = 0.6; //beta
    params[6] = 5; //K
    params[7] = 25;   //Nitermax
    int ndim = 2*NCh;
    double *min = new double[ndim];
    double *max = new double[ndim];
    double *out = new double[ndim];
    for(int i = 0; i < ndim;++i){
        min[i] = 0;
        max[i] = RyR_size*NCh;
    }
    int NCl = 1000;
    for(int numcl = 0;numcl < NCl; ++numcl){
        double res = 1e9;
        cout << "Generating cluster " << numcl << endl;
        int numtry = 1;
        while(res > 1e-9){
            cout << "try #" << numtry << endl;
            GSA gsa;
            gsa.set_parameters(params);
            gsa.optimize(f,Nthreads,ndim,min,max,out);
            res = f(out,1);
            ++numtry;
        }
        cout << "Created." << endl;
        ofstream Out("cluster_"+to_string(numcl)+".txt");
        for(int i = 0; i < ndim; i+=2)
            Out << out[i] << " " << out[i+1] << endl;
        Out.close();
        vector<channel> channels;
        for(int i = 0; i < NCh; ++i)
            channels.push_back(channel(out[2*i],out[2*i+1]));
        vector<vector<double> > results;
        vector<double> tt;
        cout << "Simulating sparks ..." << endl;
        int maxit = 2*NCh;
        for(unsigned int i = 0; i < Nscenarios; ++i)
            results.push_back(tt);
#pragma omp parallel for num_threads(Nthreads)
        for(unsigned int i = 0; i < Nscenarios; ++i){
            int numit = 1;
            results[i].push_back(0);
            vector<int> open;
            vector<channel> channels_scenario(channels);
            vector<double> rates(NCh);
            int init = MyRng.runifint(0,NCh-1);
            channels_scenario[init].isopen = true;
            open.push_back(init);
            results[i].push_back(open.size());
            while(!open.empty() && numit <= maxit){
                double alpha0 = 0;
                for(int j = 0; j < NCh;++j){
                    if(channels_scenario[j].isopen)
                        rates[j] = delta;
                    else{
                        double dist = 0;
                        for(int z = 0; z < open.size(); ++z){
                            double temp_dist = (channels_scenario[j].x -channels_scenario[open[z]].x)*(channels_scenario[j].x -channels_scenario[open[z]].x)
                                    + (channels_scenario[j].y -channels_scenario[open[z]].y)*(channels_scenario[j].y -channels_scenario[open[z]].y);
                            dist += 1/(sqrt(temp_dist));
                        }
                        dist = pow(dist,mu);
                        rates[j] = alpha*dist;
                    }
                    alpha0 += rates[j];
                }
                double time_reac = 1/alpha0 * log(1/MyRng.runif());
                results[i].push_back(results[i][results[i].size() -2] + time_reac);
                double r = MyRng.runif();
                int num = 0;
                double s = rates[num]/alpha0;
                while(s < r){
                    ++num;
                    s += rates[num]/alpha0;
                }
                channels_scenario[num].isopen = !channels_scenario[num].isopen;
                if(channels_scenario[num].isopen)
                    open.push_back(num);
                else{
                    int n = 0;
                    while(open[n] != num)
                        ++n;
                    open.erase(open.begin() + n );
                }
                results[i].push_back(open.size());
                ++numit;
            }
        }
        cout << "Done. Writing to file" << endl;
        Out.open("scenarios_"+to_string(numcl)+".txt");
        for(unsigned int i = 0; i < Nscenarios;++i){
            for(int j = 0; j < results[i].size(); ++j)
                Out << results[i][j] << " ";
            Out << endl;
        }
        Out.close();
    }
    //clear all
    delete[] params;
    delete[] min;
    delete[] max;
    delete[] out;
    return 0;
}


