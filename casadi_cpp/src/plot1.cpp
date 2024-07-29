#include "matplotlibcpp.h"
#include <cmath>
#include <iostream>
#include <vector>

namespace plt = matplotlibcpp;
using namespace std;

int main()
{
    vector<float> x = {1,3.1,4,5};
    vector<float> y = {2.1, 3.1, 4.1, 5.1};
    for(float value : x)
    {
        cout << " x "<<  value << " ";
    }
    cout << endl;
    for(float value : y)
    {
        cout << " y "<<  value << " ";
    }
    cout << endl;
    plt::plot(x, y, "r-");
    plt::grid(true);
    plt::show();

}
