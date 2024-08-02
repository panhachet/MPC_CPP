#include <iostream>
#include <tuple>
#include <cmath>
#include <ctime>
#include <map>
#include "casadi/casadi.hpp"
#include "matplotlibcpp.h"

using namespace std;
namespace ca = casadi;
namespace plt = matplotlibcpp;

ca::MX f(const ca::MX& x_, const ca::MX& u_)
{
    return vertcat(
        u_(0) * ca::MX::cos(x_(2)) - u_(1) * ca::MX::sin(x_(2)),
        u_(0) * ca::MX::sin(x_(2)) + u_(1) * ca::MX::cos(x_(2)),
        u_(2)
    );
}

ca::DM f_d(const ca::DM& x_, const ca::DM& u_)
{
    ca::DM result(3, 1);
    result(0) = u_(0) * cos(x_(2)) - u_(1) * sin(x_(2));
    result(1) = u_(0) * sin(x_(2)) + u_(1) * cos(x_(2));
    result(2) = u_(2);

    return result;
}

struct ShiftResult {
    double t;
    ca::DM st;
    ca::DM u_end;
    ca::DM x_n;
};

ShiftResult shift(double T, double t0, const ca::DM& x0, const ca::DM& u, const ca::DM& x_n, ca::DM(*f)(const ca::DM&, const ca::DM&))
{
    ca::DM f_value = f(x0, u);
    ca::DM st = x0 + T * f_value;
    ca::DM u_end = u;
    ca::DM x_n_end = x_n;
    return { t0 + T, st, u_end, x_n_end };
}

ca::DM predict_state(const ca::DM& x0, const ca::DM& u, double T, int N) {
    ca::DM states = ca::DM::zeros(N+1, 3);
    states(ca::Slice(0, 1), ca::Slice()) = x0.T();
    for (int i = 0; i < N; ++i) {
        states(i+1, 0) = states(i, 0) + (u(i, 0) * ca::DM::cos(states(i, 2)) - u(i, 1) * ca::DM::sin(states(i, 2))) * T;
        states(i+1, 1) = states(i, 1) + (u(i, 0) * ca::DM::sin(states(i, 2)) + u(i, 1) * ca::DM::cos(states(i, 2))) * T;
        states(i+1, 2) = states(i, 2) + u(i, 2) * T;
    }
    return states;
}

int main()
{
    float T = 0.2;
    int N = 30;
    float v_max = 1.0;
    float omega_max = M_PI/3.0;

    ca::Opti opti = ca::Opti();
    ca::Slice all;

    // Controls
    ca::MX u = opti.variable(3, N);
    auto vx = u(0, all);
    auto vy = u(1, all);
    auto omega = u(2, all);
    
    // States
    ca::MX X = opti.variable(3, N+1);
    auto x = X(0, all);
    auto y = X(1, all);
    auto theta = X(2, all);

    // Parameters
    ca::MX opt_x0 = opti.parameter(3);
    ca::MX opt_xs = opti.parameter(3);

    // initial condition
    opti.subject_to(X(all, 0) == opt_x0);
 
    for (int k =0; k< N; k++)
    {
        ca::MX k1 = f(X(all, k), u(all, k));
        ca::MX k2 = f(X(all, k) + T/2 * k1, u(all, k));
        ca::MX k3 = f(X(all, k) + T/2 * k2, u(all, k));
        ca::MX k4 = f(X(all, k) + T * k3, u(all, k));
        ca::MX x_next = X(all, k) + T/6 * (k1 + 2 * k2 + 2 * k3 + k4);
        opti.subject_to(X(all, k+1) == x_next);
    }

    vector<double> obs_x = {0.5, 1.0, 3.0, 3.0, 3.8, 4.5};
    vector<double> obs_y = {1.0, 1.0, 1.5, 3.0, 4.0, 4.0};
    float obs_diam = 0.3;
    float bias = 0.02;

    // Static Obstacle Constraints
    for (int i = 0; i <= N; i++) {
        for (int j = 0; j < obs_x.size(); j++) {
            ca::MX obs_constraint = ca::MX::sqrt(ca::MX::pow(X(0, i) - obs_x[j] - bias, 2) + ca::MX::pow(X(1, i) - obs_y[j] , 2)-bias) - 0.3/2.0 - obs_diam/2.0;
            opti.subject_to(opti.bounded(0.0, obs_constraint, 10));
        }
    }


    ca::DM Q = ca::DM::diag({10.0, 10.0, 1.0});
    ca::DM R = ca::DM::diag({1.0, 1.0, 0.5});
    
    ca::MX obj = 0;
    for (int i = 0; i<N; i++)
    {
        obj = obj + ca::MX::mtimes(ca::MX::mtimes((X(all, i) - opt_xs).T(), Q), (X(all, i) - opt_xs)) + ca::MX::mtimes(ca::MX::mtimes(u(all, i).T(), R), u(all, i));
    }
    opti.minimize(obj);
    
    opti.subject_to(opti.bounded(-10.0,x,10.0));
    opti.subject_to(opti.bounded(-10.0,y,10.0));
    opti.subject_to(opti.bounded(-v_max,vx,v_max));
    opti.subject_to(opti.bounded(-v_max,vy,v_max));
    opti.subject_to(opti.bounded(-omega_max,omega,omega_max));
    
    ca::Dict opts_setting;
    opts_setting["ipopt.max_iter"] = 2000;
    opts_setting["ipopt.print_level"] = 0;
    opts_setting["ipopt.acceptable_tol"] = 1e-8;
    opts_setting["ipopt.acceptable_obj_change_tol"] = 1e-6;
    opti.solver("ipopt", opts_setting);

    // Target state
    ca::DM final_state = ca::DM({6.0, 4.0, 0});
    opti.set_value(opt_xs, final_state);

    // Initial state
    int t0 = 0;
    ca::DM init_state = ca::DM::zeros(3);
    ca::DM u0 = ca::DM::zeros(3, N);

    ca::DM current_state = ca::DM::zeros(3);
    ca::DM next_state = ca::DM::zeros(3, N+1);

    vector<vector<double>> x_c;
    vector<vector<double>> u_c;
    vector<double> t_v = {t0};
    vector<vector<double>> xx;
    float sim_time = 30.0;

    int mpciter = 0;
    double start_time = static_cast<double>(clock()) / CLOCKS_PER_SEC;
    vector<double> index_t;
    while (ca::DM::norm_2(current_state - final_state).scalar() > 1e-6 && (mpciter - sim_time/T) < 0.0)
    {
        opti.set_value(opt_x0, current_state);

        //Set optimizing target with initial guess
        opti.set_initial(u, u0);
        opti.set_initial(X, next_state);
        
        //Solve the problem once again
        double t_start = static_cast<double>(clock()) / CLOCKS_PER_SEC;
        ca::OptiSol sol = opti.solve();
        index_t.push_back(static_cast<double>(clock()) / CLOCKS_PER_SEC - t_start);
        
        ca::DM u_res = sol.value(u);
        std::vector<double> control_values = {u_res(0, 0).scalar(), u_res(1, 0).scalar(), u_res(2, 0).scalar()};
        u_c.push_back(control_values);
        
        ca::DM next_states_pred = sol.value(X);
        std::vector<double> state_values = {next_states_pred(0, 0).scalar(), next_states_pred(1, 0).scalar(), next_states_pred(2, 0).scalar()};
        x_c.push_back(state_values);
        
        ShiftResult shifted_values = shift(T, t0, current_state, u_res, next_states_pred, f_d);
        
        t0 = shifted_values.t;
        current_state = shifted_values.st;
        u0 = shifted_values.u_end;
        next_state = shifted_values.x_n;
        mpciter++;
        xx.push_back({current_state(0).scalar(), current_state(1).scalar(), current_state(2).scalar()});
        vector<double> x_data, y_data, theta_data;
        for (const auto& state : xx) {
            if (state.size() == 3) {
                x_data.push_back(state[0]);
                y_data.push_back(state[1]);
                theta_data.push_back(state[2]);
            }
        }
        // cout << current_state(0).scalar() << current_state(0).scalar() << endl;
        plt::clf();
        plt::plot(x_data, y_data, "r-");

        vector<double> ix_data = {init_state(0).scalar()};
        vector<double> iy_data = {init_state(1).scalar()};
        plt::plot(ix_data, iy_data, "gx");

        vector<double> ox_data = {current_state(0).scalar()};
        vector<double> oy_data = {current_state(1).scalar()};
        plt::plot(ox_data, oy_data, "bo");

        vector<double> fx_data = {final_state(0).scalar()};
        vector<double> fy_data = {final_state(1).scalar()};
        plt::plot(fx_data, fy_data, "bx");

        double arrow_length = 0.5;
        double angle = current_state(2).scalar();
        double x_end = current_state(0).scalar() + arrow_length * cos(angle);
        double y_end = current_state(1).scalar() + arrow_length * sin(angle);

        plt::plot({current_state(0).scalar(), x_end}, {current_state(1).scalar(), y_end}, "g-"); 

        // Plot obstacle 
        plt::plot(obs_x, obs_y, "ro");

        plt::xlabel("x");
        plt::ylabel("y");
        plt::grid(true);
        plt::title("Trajectory of the Robot");
        plt::xlim(-1, 7);
        plt::ylim(-1, 7);
        plt::pause(0.01);
        
    
    
    }

    double t_v_mean = accumulate(index_t.begin(), index_t.end(), 0.0) / index_t.size();
    double total_time = (static_cast<double>(clock()) / CLOCKS_PER_SEC - start_time) / mpciter;
    cout << "Mean solving time: " << t_v_mean << " seconds." << endl;
    cout << "Total time per iteration: " << total_time << " seconds." << endl;
    
    
    plt::show();


    return 0;
}