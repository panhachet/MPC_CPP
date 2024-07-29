#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>
#include <tuple>
#include <map>
#include "casadi/casadi.hpp"
#include "matplotlibcpp.h"

using namespace std;
namespace ca = casadi;
namespace plt = matplotlibcpp;

int Q_x = 100;
int Q_y = 100;
int Q_theta = 2000;

float step_horizon = 0.1;
int N = 10;
float wheel_radius = 1;
float Lx = 0.3;
float Ly = 0.3;
int sim_time = 200;

float x_init = 0.0;
float y_init = 0.0;
float theta_init = 0.0;

float x_target = 15;
float y_target = 10;
float theta_target = M_PI / 4;

float v_max = 1;
float v_min = -1;

tuple<double, ca::DM, ca::DM> shift_timestep(double step_horizon, double t0, const ca::DM& state_init, const ca::DM& u, const ca::Function& f)
{
    ca::DM f_value = f(ca::DMVector{state_init, u(ca::Slice(), 0)})[0];
    ca::DM next_state = state_init + step_horizon * f_value;

    t0 += step_horizon;
    ca::DM u0 = ca::DM::horzcat({
        u(ca::Slice(), ca::Slice(1, u.size2())),
        ca::DM::reshape(u(ca::Slice(), -1), -1, 1)
    });

    return {t0, next_state, u0};
}

vector<double> DM2Arr(const ca::DM& dm) {
    int rows = dm.size1();
    int cols = dm.size2();
    vector<double> arr(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            arr[i * cols + j] = dm(i, j).scalar();
        }
    }
    return arr;
}

int main()
{
    // States
    ca::SX x = ca::SX::sym("x");
    ca::SX y = ca::SX::sym("y");
    ca::SX theta = ca::SX::sym("theta");
    ca::SX states = ca::SX::vertcat({x, y, theta});
    int n_states = states.numel();

    // Controls
    ca::SX V_a = ca::SX::sym("V_a");
    ca::SX V_b = ca::SX::sym("V_b");
    ca::SX V_c = ca::SX::sym("V_c");
    ca::SX V_d = ca::SX::sym("V_d");
    ca::SX controls = ca::SX::vertcat({V_a, V_b, V_c, V_d});
    int n_controls = controls.numel();

    ca::SX X = ca::SX::sym("X", n_states, N + 1);
    ca::SX U = ca::SX::sym("U", n_controls, N);
    ca::SX P = ca::SX::sym("P", n_states + n_states);

    ca::DM Q = ca::DM({
        {Q_x, 0, 0},
        {0, Q_y, 0},
        {0, 0, Q_theta}
    });
    ca::DM R = ca::DM({
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    });

    ca::SX rot_3d_z = ca::SX::vertcat({
            ca::SX::horzcat({cos(theta), -sin(theta), 0}),
            ca::SX::horzcat({sin(theta), cos(theta), 0}),
            ca::SX::horzcat({0, 0, 1})
    });

    ca::DM J = (wheel_radius/4) * ca::DM({
        {1, 1, 1, 1},
        {-1, 1, 1, -1},
        {-1/(Lx + Ly), 1/(Lx + Ly), -1/(Lx + Ly), 1/(Lx + Ly)}
    });

    ca::SX RHS = ca::SX::mtimes(rot_3d_z, ca::SX::mtimes(J, controls));

    ca::Function f = ca::Function("F", {states, controls}, {RHS});

    ca::SX cost_fn = 0;
    ca::SX g = X(ca::Slice(), 0) - P(ca::Slice(0, n_states));

    for(int k = 0; k < N; k++)
    {
        ca::SX st = X(ca::Slice(), k);
        ca::SX con = U(ca::Slice(), k);
        ca::SX st_next = X(ca::Slice(), k+1);

        // Update cost
        ca::SX st_ref = P(ca::Slice(n_states, 2*n_states));
        cost_fn = cost_fn + ca::SX::mtimes(ca::SX::mtimes((st - st_ref).T(), Q), (st - st_ref)) + ca::SX::mtimes(ca::SX::mtimes(con.T(), R), con);

        // Perform RK4
        ca::SX k1 = f(ca::SXVector{st, con})[0];
        ca::SX k2 = f(ca::SXVector{st + step_horizon/2 * k1, con})[0];
        ca::SX k3 = f(ca::SXVector{st + step_horizon/2 * k2, con})[0];
        ca::SX k4 = f(ca::SXVector{st + step_horizon * k3, con})[0];
        ca::SX st_next_RK4 = st + (step_horizon / 6 ) * (k1 + 2 * k2 + 2* k3 + k4);

        // Update constraints
        g = ca::SX::vertcat({g, st_next - st_next_RK4});
    }
    ca::SX OPT_variables = ca::SX::vertcat({
        ca::SX::reshape(X, n_states * (N+1), 1),
        ca::SX::reshape(U, n_controls * N, 1)
    });

    ca::Dict opts;
    opts["ipopt.max_iter"] = 2000;
    opts["ipopt.print_level"] = 0; 
    opts["ipopt.acceptable_tol"] = 1e-8;
    opts["ipopt.acceptable_obj_change_tol"] = 1e-6;
    opts["print_time"] = 0;

    ca::Function solver = ca::nlpsol("solver", "ipopt", {{"f", cost_fn}, {"x", OPT_variables}, {"g", g}, {"p", P}}, opts);

    ca::DM lbx = ca::DM::zeros(n_states * (N+1) + n_controls * N, 1);
    ca::DM ubx = ca::DM::zeros(n_states * (N+1) + n_controls * N, 1);
    
    lbx(ca::Slice(0, n_states*(N+1), n_states)) = -INFINITY;
    lbx(ca::Slice(1, n_states*(N+1), n_states)) = -INFINITY;
    lbx(ca::Slice(2, n_states*(N+1), n_states)) = -INFINITY;

    ubx(ca::Slice(0, n_states*(N+1), n_states)) = INFINITY;
    ubx(ca::Slice(1, n_states*(N+1), n_states)) = INFINITY;
    ubx(ca::Slice(2, n_states*(N+1), n_states)) = INFINITY;

    lbx(ca::Slice(n_states*(N+1), lbx.size1())) = v_min;
    ubx(ca::Slice(n_states*(N+1), lbx.size1())) = v_max;

    ca::DM lbg = ca::DM::zeros((n_states*(N+1)), 1);
    ca::DM ubg = ca::DM::zeros((n_states*(N+1)), 1);

    std::map<std::string, casadi::DM> args = {
        {"lbg", lbg},
        {"ubg", ubg},
        {"lbx", lbx},
        {"ubx", ubx}
    };
    
    double t0 = 0;

    ca::DM state_init = ca::DM::vertcat({x_init, y_init, theta_init});
    ca::DM state_target = ca::DM::vertcat({x_target, y_target, theta_target});
    ca::DM X0 = ca::DM::repmat(state_init, 1, N+1);
    ca::DM u0 = ca::DM::repmat(0, n_controls, N);
    int mpc_iter = 0;

    vector<double> cat_states;
    vector<double> cat_controls;
    vector<double> times;
    int num_elements_u = n_controls * N;
    int num_elements_X0 = n_states * (N + 1);

    if (num_elements_u + num_elements_X0 != 73) {
        std::cerr << "Error: Expected number of elements does not match the size of sol['x']." << std::endl;
        return -1; // Handle the error as needed
    }
        
    clock_t start_time = clock();
    while (ca::DM::norm_2(state_init - state_target).scalar() > 1e-1 && mpc_iter * step_horizon < sim_time) {
        clock_t t1 = clock();

        args["p"] = ca::SX::vertcat({state_init, state_target});
        args["x0"] = ca::SX::vertcat({ca::SX::reshape(X0, n_states * (N + 1), 1), ca::SX::reshape(u0, n_controls * N, 1)});

        map<string, ca::DM> sol = solver(args);

        // Extract and reshape the solution
        ca::DM u = ca::DM::reshape(sol["x"](ca::Slice(num_elements_X0, num_elements_X0 + num_elements_u)), n_controls, N);
        ca::DM X0 = ca::DM::reshape(sol["x"](ca::Slice(0, num_elements_X0)), n_states, N + 1);

        
        vector<double> X0_arr = DM2Arr(X0);
        vector<double> u_arr = DM2Arr(u(ca::Slice(), 0));

        cat_states.insert(cat_states.end(), X0_arr.begin(), X0_arr.end());
        cat_controls.insert(cat_controls.end(), u_arr.begin(), u_arr.end());

        clock_t t2 = clock();
        times.push_back(double(t2 - t1) / CLOCKS_PER_SEC);

        tie(t0, state_init, u0) = shift_timestep(step_horizon, t0, state_init, u, f);
        cout << mpc_iter <<endl;
        X0 = ca::SX::horzcat({X0(ca::Slice(0, X0.size1()), ca::Slice(1, X0.size2())), X0(ca::Slice(0, X0.size1()), ca::Slice(-1, X0.size2()))});
        mpc_iter++;
    }
    clock_t clost_time = clock();
    double ss_error = ca::DM::norm_2(state_init - state_target).scalar();
    cout << "Time: " << (float)(clost_time - start_time)/CLOCKS_PER_SEC << endl;
    cout << "Final error: " << ss_error << endl;


    plt::plot(cat_states);
    plt::show();
    
    return 0;
}
