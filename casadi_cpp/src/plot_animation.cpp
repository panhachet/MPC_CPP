#include <iostream>
#include "matplotlibcpp.h"
#include <cmath>

using namespace std;
namespace plt = matplotlibcpp;

int main()
{
    int n = 1000;
    vector<double> x, y ,z;
    for(int i =0; i<n; i++)
    {
        x.push_back(i*i);
        
        y.push_back(2*M_PI*i/360.0);
        z.push_back(log(i));

        if(i % 10 == 0)
        {
            plt::clf();
            plt::plot(x, y);
            plt::named_plot("log(x)", x, z);
            plt::xlim(0, n*n);
            plt::title("Sample Figure");
            plt::legend();
            plt::pause(0.01);

        }
        
    }
    plt::show();
return 0; 
}