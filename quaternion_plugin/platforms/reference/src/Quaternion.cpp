#include "Quaternion.h"
#include "openmm/Vec3.h"
#include <cstring>
#include <vector>


using namespace OpenMM;
using namespace std;

Quaternion::Quaternion(void){
    // q.zero();
    // pos1_gradients = pos2_gradients = false;

}
void Quaternion::build_correlation_matrix(const std::vector<Vec3> pos1, const std::vector<Vec3> pos2){
    
    //C = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    
    memset(C, 0, sizeof(C));
    
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        for (int k = 0; k < pos1.size(); k++) {
            C[i][j] += pos1[k][i]*pos2[k][j];
        }

}
void Quaternion::calculate_overlap_matrix(void){

    S = Array2D<double>(4, 4);

    S[0][0] =  - C[0][0] - C[1][1]- C[2][2];
    S[1][1] = - C[0][0] + C[1][1] + C[2][2];
    S[2][2] =  C[0][0] - C[1][1] + C[2][2];
    S[3][3] =  C[0][0] + C[1][1] - C[2][2];
    S[0][1] = - C[1][2] + C[2][1];
    S[0][2] = C[0][2] - C[2][0];
    S[0][3] = - C[0][1] + C[1][0];
    S[1][2] = - C[0][1] - C[1][0];
    S[1][3] = - C[0][2] - C[2][0];
    S[2][3] = - C[1][2] - C[2][1];
    S[1][0] = S[0][1];
    S[2][0] = S[0][2];
    S[2][1] = S[1][2];
    S[3][0] = S[0][3];
    S[3][1] = S[1][3];
    S[3][2] = S[2][3];

}
void  Quaternion::diagonalize_matrix(const std::vector<double> normquat)
{
    JAMA::Eigenvalue<double> eigen(S);
    eigen.getRealEigenvalues(S_eigval);
    Array2D<double> vectors;
    eigen.getV(S_eigvec);
    double dot;
    //Normalise each eigenvector in the direction closer to norm
    for (unsigned i=0;i<4;i++) {
        dot=0.0;
        for (unsigned j=0;j<4;j++) {
            dot += normquat[j] * S_eigvec[i][j];
        }
        if (dot < 0.0)
            for (unsigned j=0;j<4;j++)
                S_eigvec[i][j] =- S_eigvec[i][j];
    }

}

std::vector<double> Quaternion::getQfromEigenvecs(unsigned idx){
    return {S_eigvec[idx][0], S_eigvec[idx][1], S_eigvec[idx][2], S_eigvec[idx][3]};
}

void Quaternion::request_group1_gradients(unsigned n){
    dS_1.resize(n, Array2D<Vec3>(4, 4));
    dL0_1.resize(n, Vec3(0.0, 0.0, 0.0));
    dQ0_1.resize(n, vector<Vec3>(4));
    pos1_gradients = true;
}

void Quaternion::request_group2_gradients(unsigned n){
    dS_2.resize(n, Array2D<Vec3>(4, 4));
    dL0_2.resize(n, Vec3(0.0, 0.0, 0.0));
    dQ0_2.resize(n, vector<Vec3>(4));
    pos2_gradients = true;
}

// From NAMD
void Quaternion::calc_optimal_rotation(const std::vector<OpenMM::Vec3> pos1, const std::vector<OpenMM::Vec3> pos2, const std::vector<double> normquat){

    q.assign(4, 0);
    build_correlation_matrix(pos1, pos2);
    calculate_overlap_matrix();
    diagonalize_matrix(normquat);

    double const L0 = S_eigval[0];
    double const L1 = S_eigval[1];
    double const L2 = S_eigval[2];
    double const L3 = S_eigval[3];

    std::vector<double> const Q0 = getQfromEigenvecs(0);
    std::vector<double> const Q1 = getQfromEigenvecs(1);
    std::vector<double> const Q2 = getQfromEigenvecs(2);
    std::vector<double> const Q3 = getQfromEigenvecs(3);

    lambda = L0;
    q = Q0;

    q0 = q[0];  q1 = q[1]; q2 = q[2]; q3 = q[3];

    if (pos1_gradients){
    for (unsigned ia=0; ia < dS_1.size(); ia++) {
        //if (refw1[ia]==0) continue; //Only apply forces to weighted atoms in the RMSD calculation.

        double const rx = pos2[ia][0];
        double const ry = pos2[ia][1];
        double const rz = pos2[ia][2];

        Array2D<Vec3> &ds_1 = dS_1[ia];

        ds_1[0][0] = Vec3(  rx,  ry,  rz);
        ds_1[1][0] = Vec3( 0.0, -rz,  ry);
        ds_1[0][1] = ds_1[1][0];
        ds_1[2][0] = Vec3(  rz, 0.0, -rx);
        ds_1[0][2] = ds_1[2][0];
        ds_1[3][0] = Vec3( -ry,  rx, 0.0);
        ds_1[0][3] = ds_1[3][0];
        ds_1[1][1] = Vec3(  rx, -ry, -rz);
        ds_1[2][1] = Vec3(  ry,  rx, 0.0);
        ds_1[1][2] = ds_1[2][1];
        ds_1[3][1] = Vec3(  rz, 0.0,  rx);
        ds_1[1][3] = ds_1[3][1];
        ds_1[2][2] = Vec3( -rx,  ry, -rz);
        ds_1[3][2] = Vec3( 0.0,  rz,  ry);
        ds_1[2][3] = ds_1[3][2];
        ds_1[3][3] = Vec3( -rx, -ry,  rz);

        Vec3               &dl0_1 = dL0_1[ia];
        vector<Vec3>        &dq0_1 = dQ0_1[ia];

        for (unsigned i = 0; i < 4; i++) {
            for (unsigned j = 0; j < 4; j++) {
                dl0_1 += -1 * (Q0[i] * ds_1[i][j] * Q0[j]);
            }
        }
        for (unsigned p=0; p<4; p++) {
            for (unsigned i=0 ;i<4; i++) {
                for (unsigned j=0; j<4; j++) {
                    dq0_1[p] += -1 * (
                            (Q1[i] * ds_1[i][j] * Q0[j]) / (L0-L1) * Q1[p] +
                            (Q2[i] * ds_1[i][j] * Q0[j]) / (L0-L2) * Q2[p] +
                            (Q3[i] * ds_1[i][j] * Q0[j]) / (L0-L3) * Q3[p]);
                    }
                }
            }
        } // First loop

    }
    if (pos2_gradients) {
    for (unsigned ia=0; ia < dS_2.size(); ia++) {
        //if (refw1[ia]==0) continue; //Only apply forces to weighted atoms in the RMSD calculation.

        double const rx = pos1[ia][0];
        double const ry = pos1[ia][1];
        double const rz = pos1[ia][2];

        Array2D<Vec3> &ds_2 = dS_2[ia];

        ds_2[0][0] = Vec3(  rx,  ry,  rz);
        ds_2[1][0] = Vec3( 0.0, -rz,  ry);
        ds_2[0][1] = ds_2[1][0];
        ds_2[2][0] = Vec3(  rz, 0.0, -rx);
        ds_2[0][2] = ds_2[2][0];
        ds_2[3][0] = Vec3( -ry,  rx, 0.0);
        ds_2[0][3] = ds_2[3][0];
        ds_2[1][1] = Vec3(  rx, -ry, -rz);
        ds_2[2][1] = Vec3(  ry,  rx, 0.0);
        ds_2[1][2] = ds_2[2][1];
        ds_2[3][1] = Vec3(  rz, 0.0,  rx);
        ds_2[1][3] = ds_2[3][1];
        ds_2[2][2] = Vec3( -rx,  ry, -rz);
        ds_2[3][2] = Vec3( 0.0,  rz,  ry);
        ds_2[2][3] = ds_2[3][2];
        ds_2[3][3] = Vec3( -rx, -ry,  rz);

        Vec3                &dl0_2 = dL0_2[ia];
        vector<Vec3>        &dq0_2 = dQ0_2[ia];

        for (unsigned i = 0; i < 4; i++) {
            for (unsigned j = 0; j < 4; j++) {
                dl0_2 += -1 * (Q0[i] * ds_2[i][j] * Q0[j]);
            }
        }
        for (unsigned p=0; p<4; p++) {
            for (unsigned i=0 ;i<4; i++) {
                for (unsigned j=0; j<4; j++) {
                    dq0_2[p] += -1 * (
                            (Q1[i] * ds_2[i][j] * Q0[j]) / (L0-L1) * Q1[p] +
                            (Q2[i] * ds_2[i][j] * Q0[j]) / (L0-L2) * Q2[p] +
                            (Q3[i] * ds_2[i][j] * Q0[j]) / (L0-L3) * Q3[p]);
                    }
                }
            }
        } // Second loop
    }

    }

// From NAMD
std::vector<double> Quaternion::position_derivative_inner(const Vec3 &pos, const Vec3 &vec)
{
  std::vector<double> result(4, 0);


  result[0] =   2.0 * pos[0] * q0 * vec[0]
               +2.0 * pos[1] * q0 * vec[1]
               +2.0 * pos[2] * q0 * vec[2]

               -2.0 * pos[1] * q3 * vec[0]
               +2.0 * pos[2] * q2 * vec[0]

               +2.0 * pos[0] * q3 * vec[1]
               -2.0 * pos[2] * q1 * vec[1]

               -2.0 * pos[0] * q2 * vec[2]
               +2.0 * pos[1] * q1 * vec[2];


  result[1] =  +2.0 * pos[0] * q1 * vec[0]
               -2.0 * pos[1] * q1 * vec[1]
               -2.0 * pos[2] * q1 * vec[2]

               +2.0 * pos[1] * q2 * vec[0]
               +2.0 * pos[2] * q3 * vec[0]

               +2.0 * pos[0] * q2 * vec[1]
               -2.0 * pos[2] * q0 * vec[1]

               +2.0 * pos[0] * q3 * vec[2]
               +2.0 * pos[1] * q0 * vec[2];


  result[2] =  -2.0 * pos[0] * q2 * vec[0]
               +2.0 * pos[1] * q2 * vec[1]
               -2.0 * pos[2] * q2 * vec[2]

               +2.0 * pos[1] * q1 * vec[0]
               +2.0 * pos[2] * q0 * vec[0]

               +2.0 * pos[0] * q1 * vec[1]
               +2.0 * pos[2] * q3 * vec[1]

               -2.0 * pos[0] * q0 * vec[2]
               +2.0 * pos[1] * q3 * vec[2];


  result[3] =  -2.0 * pos[0] * q3 * vec[0]
               -2.0 * pos[1] * q3 * vec[1]
               +2.0 * pos[2] * q3 * vec[2]

               -2.0 * pos[1] * q0 * vec[0]
               +2.0 * pos[2] * q1 * vec[0]

               +2.0 * pos[0] * q0 * vec[1]
               +2.0 * pos[2] * q2 * vec[1]

               +2.0 * pos[0] * q1 * vec[2]
               +2.0 * pos[1] * q2 * vec[2];

  return result;
}

// std::vector<double> Quaternion::conjugat(void){

//     return std::vector<double>(q0, -q1, -q2, -q3);
// }
// Vec3 Quaternion::rotate(const Vec3 vec){

//     std::vector<double> qc, vec4d, result;
//     vec4d = std::vector<double>(0.0, vec[0], vec[1], vec[2]);
//     result = std::vector<double>(0.0, 0.0, 0.0, 0.0);
//     qc = this->conjugat();
//     for (unsigned i=0; i<4; i++){
//         result[i] = q[i] * vec[i] * qc[i];
//     }
//     return Vec3(result[1], result[2], result[3]);
// }

std::vector<Vec3> Quaternion::rotateCoordinates(std::vector<double> qr, const std::vector<Vec3> pos){
    std::vector<Vec3> rot_pos;
    unsigned ntot = pos.size();
    //rot_pos.resize(ntot, Vec3(0,0,0));
    for (unsigned i=0; i<ntot; i++)
            rot_pos.push_back((this->quaternionRotate(qr, pos[i])));

    return rot_pos;
}

std::vector<double> Quaternion::quaternionInvert(const std::vector<double> q){
    return {q[0],-q[1],-q[2],-q[3]};
}


std::vector<double> Quaternion::quaternionProduct(const std::vector<double>& v1,const std::vector<double>& v2){
    return {
        v1[0]*v2[0]-v1[1]*v2[1]-v1[2]*v2[2]-v1[3]*v2[3],
        v1[0]*v2[1]+v1[1]*v2[0]+v1[2]*v2[3]-v1[3]*v2[2],
        v1[0]*v2[2]+v1[2]*v2[0]+v1[3]*v2[1]-v1[1]*v2[3],
        v1[0]*v2[3]+v1[3]*v2[0]+v1[1]*v2[2]-v1[2]*v2[1]};
}


Vec3 Quaternion::quaternionRotate(const std::vector<double>& qq, const Vec3& v){
    double q0 = qq[0];
    Vec3 vq = Vec3(qq[1],qq[2],qq[3]);
    Vec3 a;
    Vec3 b;
    a = vq.cross(v) + q0 * v;
    b = vq.cross(a);
    return b+b+v;
}


