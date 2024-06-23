#include <iostream>
#include <chrono>
#include <sciplot/sciplot.hpp>
#include <Eigen/Dense>
#include <memory>
#include "numericlibs/RK4.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using Numericlib::System;
using Numericlib::RK4;
using Numericlib::SolnObject;

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

int main(void){
    VectorXd x0(2);
    x0 << 0.0, 1.0;
    VectorXd params(2);
    params << 9.8, 1.0;

    auto start = high_resolution_clock::now();
    std::shared_ptr<System> pendulum = std::make_shared<SimplePendulum>();
    SolnObject soln = RK4(pendulum, x0, params, 1e-4, 10.0);
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