//
// Created by serik1987 on 20.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_LINEARFIT_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_LINEARFIT_H

#include <vector>
#include <iostream>
#include <algorithm>
#include "../exceptions.h"

namespace GLOBAL_NAMESPACE {

    class LinearFit {
    private:
        int dim;
        int N;

        std::vector<double> Sy;
        std::vector<double> Sxy;
        double S;
        double Sx;
        double Sxx;
        double DD;
        std::vector<double> slopes;
        std::vector<double> intersects;
    public:
        explicit LinearFit(int dim);
        void add(const double* values);
        void ready();

        friend std::ostream& operator<<(std::ostream& out, const LinearFit& fit);

        [[nodiscard]] double getIntersect(int chan) { return intersects[chan]; }
        [[nodiscard]] double getSlope(int chan) { return slopes[chan]; }

    };

    std::ostream& operator<<(std::ostream& out, const std::vector<double>& vector);

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_LINEARFIT_H
