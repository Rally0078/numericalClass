//Rally's ODE solver library
//Currently supports RK4
#pragma once

#include <cmath>
#include <Eigen/Dense>
#include <memory>

using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace Numericlib {
    //Base class for all ODE systems. To solve an ODE system, create a derived class of this class, and overload the operator().
    //Overloaded operator() must return a VectorXd containing the list of values evaluated by the ODE at a given time.
    class System {
        public:
        System() {};
        //Overloaded operator(). Implement the virtual method in the derived classes with the appropriate return values.
        //@param x0 contains the current values of the system
        //@param t contains the current time
        //@param params contains the parameters of the ODE
        //@return returns a VectorXd containing the values of the ODE evaluated at that time with the given current values.
        virtual VectorXd operator() (VectorXd x0, double t, VectorXd params) = 0;
    };
    /*Structure containing solution of the ODE system.
    @param result contains the results of the ODE solution.
    @param timeArray contains each of the time steps in the ODE solution.
    */
    struct SolnObject{
        MatrixXd result;
        VectorXd timeArray;
    };
    /*The RK4 solver
    Params:
    @param system contains the ODE system in the form of a shared_ptr<System> pointing to an instance of a derived class of System.
    @param x0 contains the initial values of the ODE system.
    @param params contains the parameters of the ODE system.
    @param h contains the time step size.
    @param tFin contains the final time of the solution. By default, initial time is taken to be 0.0

    @return SolnObject containing the ODE's solution and the time steps.
    */
    SolnObject RK4(std::shared_ptr<System> system, VectorXd x0, VectorXd params, double h=0.01, double tFin = 10.0){
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
}