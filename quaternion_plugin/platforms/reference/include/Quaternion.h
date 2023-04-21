#ifndef QUATERNION_H
#define QUATERNION_H


#include "openmm/Vec3.h"
#include "jama_eig.h"
#include <vector>

using namespace OpenMM;
using namespace std;


class Quaternion {
    public:
        Array2D<double> S, S_eigvec;
        //Array2D<double>  S(4, 4); //S_eigvec(4, 4);
        double C[3][3]; 
        Array1D< double > S_eigval;
        std::vector<double> q = vector<double>(4);
        double R;
        double lambda;
        double q0, q1, q2, q3;

        /// Derivatives of S
        std::vector<Array2D<Vec3>> dS_1,  dS_2;
        /// Derivatives of leading eigenvalue
        std::vector<Vec3>  dL0_1, dL0_2;
        /// Derivatives of leading eigenvector
        std::vector<std::vector<Vec3>> dQ0_1, dQ0_2;
        Quaternion(void);
        void calc_optimal_rotation(const std::vector<OpenMM::Vec3> pos1, const std::vector<Vec3> pos2, const std::vector<double> normquat);
        void build_correlation_matrix(const std::vector<Vec3> pos1, const std::vector<Vec3> pos2);
        void calculate_overlap_matrix(void);
        void diagonalize_matrix(const std::vector<double> normquat);
        std::vector<double> getQfromEigenvecs(unsigned idx);
        void request_group1_gradients(unsigned n);
        void request_group2_gradients(unsigned n);
        std::vector<double> position_derivative_inner(const Vec3 &pos, const Vec3 &vec);
        //std::vector<double> conjugat(void);
        Vec3 rotate(const Vec3 vec);
        std::vector<Vec3> rotateCoordinates(const std::vector<double>qr, const std::vector<Vec3> pos);
        std::vector<double> quaternionInvert(const std::vector<double>q);
        std::vector<double> quaternionProduct(const std::vector<double>& v1, const std::vector<double>& v2);
        Vec3 quaternionRotate(const std::vector<double>& qq, const Vec3& v);
    private:
        bool pos1_gradients, pos2_gradients;

    //void calculate_gradients(const std::vector<Vec3> pos);

};


#endif // QUATERNION_H
