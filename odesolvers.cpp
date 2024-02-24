#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <chrono>
#include <sciplot/sciplot.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

class System {

    public:
    System() {};
    virtual VectorXd operator() (VectorXd x0, double t, VectorXd params) = 0;
};


class SimplePendulum : public System{
    public:
    SimplePendulum() {};
    VectorXd operator()(VectorXd x0, double t, VectorXd params) override {
        double g = params[0];
        double l1 = params[1];

        double theta = x0[0];
        double thetadot = x0[1];
        double thetaddot = -g/l1 * std::sin(theta);

        VectorXd result(2);
        result[0] = thetadot;
        result[1] = thetaddot;

        return result;
    }
};

struct SolnObject{
    MatrixXd result;
    VectorXd timeArray;
};

SolnObject RK4(System* system, VectorXd x0, VectorXd params, double h=0.01, double tFin = 10.0){
    double t = 0;
    long N = (tFin + h)/h;
    MatrixXd result(x0.rows(), N);
    VectorXd timeArray(N);
    long i = 0;
    result.col(i) = x0;
    timeArray(i) = 0.0;
    while(t < tFin){
        VectorXd k1 = (*system)(x0, t, params);
        VectorXd k2 = (*system)(x0 + k1 * h/2.0, t + h/2.0, params);
        VectorXd k3 = (*system)(x0 + k2 * h/2.0, t + h/2.0, params);
        VectorXd k4 = (*system)(x0 + k3 * h, t + h, params);
        x0 = x0 + h/6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        i = i + 1;
        t = t + h;
        if(i < N){ 
            result.col(i) = x0;
            timeArray(i) = t;
        }
        else
            break;
    }
    return SolnObject{result, timeArray};
}

int main(void){
    VectorXd x0(2);
    x0 << 0.0, 1.0;
    VectorXd params(2);
    params << 9.8, 1.0;

    auto start = high_resolution_clock::now();
    System *pendulum = new SimplePendulum();
    SolnObject soln = RK4(pendulum, x0, params, 1e-4, 10.0);
    delete pendulum;
    auto stop = high_resolution_clock::now();
    auto diff = duration_cast<milliseconds> (stop-start);
    std::cout<<"Time taken = "<< diff.count()/1000.0<<" seconds";

    sciplot::Plot2D plot;

    plot.xlabel("Time (s)");
    plot.ylabel("Amplitude");
    plot.xrange(0.0, 12.0);
    plot.yrange(-1.0, 1.0);
    plot.legend().atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
    plot.drawCurve(soln.timeArray, soln.result.row(0)).label("Amplitude vs time");

    sciplot::Figure fig = {{plot}};
    sciplot::Canvas canvas = {{fig}};
    canvas.size(1024, 768);
    canvas.show();
}