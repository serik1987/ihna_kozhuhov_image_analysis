//
// Created by serik1987 on 20.01.2020.
//

#include "LinearFit.h"

namespace GLOBAL_NAMESPACE {

    LinearFit::LinearFit(int dim): N(0), dim(dim), Sy(dim), Sxy(dim), slopes(dim), intersects(dim){
        std::fill(Sy.begin(), Sy.end(), 0.0);
        std::fill(Sxy.begin(), Sxy.end(), 0.0);
        S = Sx = Sxx = DD = 0.0;
    }

    std::ostream &operator<<(std::ostream &out, const LinearFit &fit) {
        out << "Sy = " << fit.Sy << std::endl;
        out << "Sxy = " << fit.Sxy << std::endl;
        out << "S = " << fit.S << std::endl;
        out << "Sx = " << fit.Sx << std::endl;
        out << "Sxx = " << fit.Sxx << std::endl;
        out << "DD = " << fit.DD << std::endl;
        out << "slopes = " << fit.slopes << std::endl;
        out << "intersects = " << fit.intersects << std::endl;

        return out;
    }

    void LinearFit::add(const double *values) {
        for (int i=0; i < dim; ++i){
            Sy[i] += values[i];
            Sxy[i] += values[i] * N;
        }
        N++;
    }

    void LinearFit::ready() {
        S = (double)N;
        Sx = 0.5 * S * (S-1.0);
        Sxx = Sx * (2.0 * S - 1.0) / 3.0;
        DD = S * Sxx - Sx * Sx;

        for (int i=0; i < dim; ++i){
            slopes[i] = (S * Sxy[i] - Sx * Sy[i]) / DD;
            intersects[i] =(Sxx * Sy[i] - Sx * Sxy[i]) / DD;
        }
    }

    std::ostream &operator<<(std::ostream &out, const std::vector<double>& vector) {
        std::for_each(vector.begin(), vector.end(), [&out](double x) {
            out << x << "\t";
        });

        return out;
    }
}