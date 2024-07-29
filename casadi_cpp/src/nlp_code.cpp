#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <casadi/casadi.hpp>

using namespace std;
namespace fs = filesystem;
namespace ca = casadi;
int main()
{
    // Optimization variable
    ca::MX x = ca::MX::sym("x", 2);

    // Objective Function
    ca::MX f = x(0)*x(0) + x(1)*x(1);

    // Constrains 

    ca::MX g = x(0) + x(1) - 10;

    // Create an NLP solver 
    ca::Function solver = ca::nlpsol("solver", "ipopt", {{"x", x}, {"f", f}, {"g", g}});

    // Initial guess for the solution
    map<string, ca::DM> arg, res;
    arg["lbx"] = -ca::DM::inf();
    arg["ubx"] = ca::DM::inf();
    arg["lbg"] = 0;
    arg["ubg"] = ca::DM::inf();
    arg["x0"] = 0;

    
    // Solve the NLP
    res = solver(arg);

    // Print out the solution

    cout << "=======" << endl;
    cout << "objective at solution" << res.at("f") << endl;
    cout << "primal solution" << res.at("x") << endl;

    return 0;

}

