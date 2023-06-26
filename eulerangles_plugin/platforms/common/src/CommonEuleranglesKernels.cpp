/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014-2021 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CommonEuleranglesKernels.h"
#include "CommonEuleranglesKernelSources.h"
#include "openmm/common/BondedUtilities.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include "jama_eig.h"
#include <set>
#include <cstring>

using namespace EuleranglesPlugin;
using namespace OpenMM;
using namespace std;

template <class REAL>
void calculateQRotation(std::vector<REAL> C, vector<mm_double4>& eigvec_buffer, vector<REAL>& eigval_buffer, vector<double>& q){
    double C2[3][3];
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        {
         C2[i][j] = C[3*i+j];
        }

    Array2D<double> S = Array2D<double>(4, 4);
    S[0][0] =  - C2[0][0] - C2[1][1]- C2[2][2];
    S[1][1] = - C2[0][0] + C2[1][1] + C2[2][2];
    S[2][2] =  C2[0][0] - C2[1][1] + C2[2][2];
    S[3][3] =  C2[0][0] + C2[1][1] - C2[2][2];
    S[0][1] = - C2[1][2] + C2[2][1];
    S[0][2] = C2[0][2] - C2[2][0];
    S[0][3] = - C2[0][1] + C2[1][0];
    S[1][2] = - C2[0][1] - C2[1][0];
    S[1][3] = - C2[0][2] - C2[2][0];
    S[2][3] = - C2[1][2] - C2[2][1];
    S[1][0] = S[0][1];
    S[2][0] = S[0][2];
    S[2][1] = S[1][2];
    S[3][0] = S[0][3];
    S[3][1] = S[1][3];
    S[3][2] = S[2][3];

    Array2D<double> S_eigvec;
    Array1D<double> S_eigval;
    JAMA::Eigenvalue<double> eigen(S);
    eigen.getRealEigenvalues(S_eigval);
    eigen.getV(S_eigvec);
    double dot;
    std::vector<double> normquat = {1.0, 0.0, 0.0, 0.0};

    // transpose
    Array2D<double> temp = Array2D<double>(4, 4);
    for (int i=0;i<4;i++) {
        for (int j=0;j<4;j++)
                temp[j][i] = S_eigvec[i][j];
    }

    S_eigvec = temp;

   // Normalise each eigenvector in the direction closer to norm
    for (int i = 0; i<4; i++) {
        dot=0.0;
        for (int j=0;j<4;j++) {
            dot += normquat[j] * S_eigvec[i][j];
        }
        if (dot < 0.0)
            for (int j=0; j<4; j++)
                S_eigvec[i][j] = -S_eigvec[i][j];
    }

    for (int i=0; i<4; i++)
        q.push_back(S_eigvec[0][i]);
    eigval_buffer = {static_cast<REAL>(S_eigval[0]), static_cast<REAL>(S_eigval[1]),
                                static_cast<REAL>(S_eigval[2]), static_cast<REAL>(S_eigval[3])};

    for (int i=0;i<4;i++)
        eigvec_buffer.push_back(mm_double4(S_eigvec[i][0], S_eigvec[i][1], S_eigvec[i][2], S_eigvec[i][3]));
}

double asinDerivative(double x) {
    return 1 / sqrt(1 - x * x);
}

void atan2Derivatives(double x, double y, double& dAtan2_dx, double& dAtan2_dy) {
    double atan2_xy = std::atan2(x, y);
    double denominator = x * x + y * y;

    // Derivative with respect to x: d(atan2(x, y))/dx = y / (x^2 + y^2)
    dAtan2_dx = y / denominator;

    // Derivative with respect to y: d(atan2(x, y))/dy = -x / (x^2 + y^2)
    dAtan2_dy = -x / denominator;
}

template <class REAL>
void calculateDeriv(std::vector<double> q, std::string angle, vector<REAL>& anglederiv_buffer, double& energy){
    double radian_to_degree = 57.2958; //180 / 3.1415926;
    double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3];
    if (angle == "Theta") {
        double x = 2 * (q1 * q3 - q4 * q2);
        energy = radian_to_degree * asin(x);
        double deriv = 2 * radian_to_degree * asinDerivative(x);
        anglederiv_buffer[0] = deriv * q3;
        anglederiv_buffer[1] = -deriv * q4;
        anglederiv_buffer[2] = deriv * q1;
        anglederiv_buffer[3] = -deriv * q2;

    }
    else if (angle == "Phi"){
        double x = 2*(q1*q2+q3*q4), y = 1-2*(q2*q2+q3*q3);
        energy = radian_to_degree * atan2(x, y);
        double deriv_x, deriv_y;
        atan2Derivatives(x, y, deriv_x, deriv_y);
        anglederiv_buffer[0] = static_cast<REAL>(2 * radian_to_degree * q2 * deriv_x);  // dE/dq1
        anglederiv_buffer[1] = static_cast<REAL>(2 * radian_to_degree * (q1 * deriv_x - 2 * q2 * deriv_y));  // dE/dq2
        anglederiv_buffer[2] = static_cast<REAL>(2 * radian_to_degree * (q4 * deriv_x - 2 * q3 * deriv_y)); // dE/dq3
        anglederiv_buffer[3] = static_cast<REAL>(2 * radian_to_degree * q3 * deriv_x); // dE/dq4
    }
    else if (angle == "Psi"){
        double x = 2*(q1*q4+q2*q3), y = 1-2*(q3*q3+q4*q4);
        energy = radian_to_degree * atan2(x, y);
        double deriv_x, deriv_y;
        atan2Derivatives(x, y, deriv_x, deriv_y);
        anglederiv_buffer[0] = static_cast<REAL>(2 * radian_to_degree * q4 * deriv_x);  // dE/dq1
        anglederiv_buffer[1] = static_cast<REAL>(2 * radian_to_degree * q3 * deriv_x);  // dE/dq2
        anglederiv_buffer[2] = static_cast<REAL>(2 * radian_to_degree * (q2 * deriv_x - 2 * q3 * deriv_y));  // dE/dq3
        anglederiv_buffer[3] = static_cast<REAL>(2 * radian_to_degree * (q1 * deriv_x - 2 * q4 * deriv_y)); // dE/dq4
    }
    else
        throw OpenMMException("updateParametersInContext: The angle type is not correct");
}

void calculateDeriv_cpu(std::vector<double> q, std::string angle, std::vector<double>& deriv_const, double& energy){
    double radian_to_degree = 180 / 3.1415926;
    double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3];
    if (angle == "Theta") {
        double x = 2 * (q1 * q3 - q4 * q2);
        energy = radian_to_degree * asin(x);
        double deriv = 2 * radian_to_degree * asinDerivative(x);
        deriv_const[0] = deriv * q3;
        deriv_const[1] = -deriv * q4;
        deriv_const[2] = deriv * q1;
        deriv_const[3] = -deriv * q2;
    }
    else if (angle == "Phi"){
        double x = 2*(q1*q2+q3*q4), y = 1-2*(q2*q2+q3*q3);
        energy = radian_to_degree * atan2(x, y);
        double deriv_x, deriv_y;
        atan2Derivatives(x, y, deriv_x, deriv_y);
        deriv_const[0] = 2 * radian_to_degree * q2 * deriv_x;  // dE/dq1
        deriv_const[1] = 2 * radian_to_degree * (q1 * deriv_x - 2 * q2 * deriv_y);  // dE/dq2
        deriv_const[2] = 2 * radian_to_degree * (q4 * deriv_x - 2 * q3 * deriv_y); // dE/dq3
        deriv_const[3] = 2 * radian_to_degree * q3 * deriv_x; // dE/dq4
    }
    else if (angle == "Psi"){
        double x = 2*(q1*q4+q2*q3), y = 1-2*(q3*q3+q4*q4);
        energy = radian_to_degree * atan2(x, y);
        double deriv_x, deriv_y;
        atan2Derivatives(x, y, deriv_x, deriv_y);
        deriv_const[0] = 2 * radian_to_degree * q4 * deriv_x;  // dE/dq1
        deriv_const[1] = 2 * radian_to_degree * q3 * deriv_x;  // dE/dq2
        deriv_const[2] = 2 * radian_to_degree * (q2 * deriv_x - 2 * q3 * deriv_y);  // dE/dq3
        deriv_const[3] =  2 * radian_to_degree * (q1 * deriv_x - 2 * q4 * deriv_y); // dE/dq4
    }
    else
         throw OpenMMException("updateParametersInContext: The angle type is not correct");

}
Vec3 calculateCOG(const std::vector<Vec3> pos, vector<int> particles){
    vector<Vec3> pos2;
    for (int i : particles){
        pos2.push_back(pos[i]);
    }
    Vec3 center = Vec3(0.0, 0.0, 0.0);
    for (Vec3 p : pos2)
        center += p;

    center /= pos2.size();
    return center;
}

Vec3 quaternionRotate(const std::vector<double>& qq, const Vec3& v){
    double q0 = qq[0];
    Vec3 vq = Vec3(qq[1],qq[2],qq[3]);
    Vec3 a;
    Vec3 b;
    a = v.cross(vq) + q0 * v;
    b = a.cross(vq);
    return b+b+v;
}

template <class REAL>
void calcForces(std::vector<REAL> C, const std::vector<OpenMM::Vec3> refpos, vector<int> particles, std::string angle,
                vector<mm_double4>& forces_buffer, double& energy, bool rotate=false,
                vector<double> qrot={0.0, 0.0, 0.0, 0.0}){


    double C2[3][3];
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        {
         C2[i][j] = C[3*i+j];
        }

    Array2D<double> S = Array2D<double>(4, 4);
    S[0][0] =  - C2[0][0] - C2[1][1]- C2[2][2];
    S[1][1] = - C2[0][0] + C2[1][1] + C2[2][2];
    S[2][2] =  C2[0][0] - C2[1][1] + C2[2][2];
    S[3][3] =  C2[0][0] + C2[1][1] - C2[2][2];
    S[0][1] = - C2[1][2] + C2[2][1];
    S[0][2] = C2[0][2] - C2[2][0];
    S[0][3] = - C2[0][1] + C2[1][0];
    S[1][2] = - C2[0][1] - C2[1][0];
    S[1][3] = - C2[0][2] - C2[2][0];
    S[2][3] = - C2[1][2] - C2[2][1];
    S[1][0] = S[0][1];
    S[2][0] = S[0][2];
    S[2][1] = S[1][2];
    S[3][0] = S[0][3];
    S[3][1] = S[1][3];
    S[3][2] = S[2][3];
    Array2D<double> S_eigvec;
    Array1D<double> S_eigval;
    JAMA::Eigenvalue<double> eigen(S);
    eigen.getRealEigenvalues(S_eigval);
    eigen.getV(S_eigvec);
    double dot;
    std::vector<double> normquat = {1.0, 0.0, 0.0, 0.0};

    // transpose
    Array2D<double> temp = Array2D<double>(4, 4);
    for (int i=0;i<4;i++) {
        for (int j=0;j<4;j++)
                temp[j][i] = S_eigvec[i][j];
    }

    S_eigvec = temp;

   // Normalise each eigenvector in the direction closer to norm
    for (int i = 0; i<4; i++) {
        dot=0.0;
        for (int j=0;j<4;j++) {
            dot += normquat[j] * S_eigvec[i][j];
        }
        if (dot < 0.0)
            for (int j=0; j<4; j++)
                S_eigvec[i][j] = -S_eigvec[i][j];
    }

    double const L0 = S_eigval[0];
    double const L1 = S_eigval[1];
    double const L2 = S_eigval[2];
    double const L3 = S_eigval[3];

    std::vector<double> const Q0 = {S_eigvec[0][0], S_eigvec[0][1], S_eigvec[0][2], S_eigvec[0][3]};
    std::vector<double> const Q1 = {S_eigvec[1][0], S_eigvec[1][1], S_eigvec[1][2], S_eigvec[1][3]};
    std::vector<double> const Q2 = {S_eigvec[2][0], S_eigvec[2][1], S_eigvec[2][2], S_eigvec[2][3]};
    std::vector<double> const Q3 = {S_eigvec[3][0], S_eigvec[3][1], S_eigvec[3][2], S_eigvec[3][3]};

    std::vector<double> q = Q0;
    vector<double> deriv_const(4);
    calculateDeriv_cpu(q, angle, deriv_const, energy);
    int numParticles = particles.size();
    for (unsigned ia=0; ia < numParticles; ia++) {
        //if (refw1[ia]==0) continue; //Only apply forces to weighted atoms in the RMSD calculation.

        double const rx = refpos[particles[ia]][0];
        double const ry = refpos[particles[ia]][1];
        double const rz = refpos[particles[ia]][2];

        Array2D<Vec3> ds_1 = Array2D<Vec3>(4, 4);

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

        vector<Vec3>  dq0_1;

        for (unsigned p=0; p<4; p++) {
            dq0_1.push_back(Vec3(0.0, 0.0, 0.0));
            for (unsigned i=0 ;i<4; i++) {
                for (unsigned j=0; j<4; j++) {
                    dq0_1[p] += -1 * ((Q1[i] * ds_1[i][j] * Q0[j]) / (L0-L1) * Q1[p]
                                    + (Q2[i] * ds_1[i][j] * Q0[j]) / (L0-L2) * Q2[p]
                                    + (Q3[i] * ds_1[i][j] * Q0[j]) / (L0-L3) * Q3[p]);
                }
            }
        }
        Vec3 force = Vec3(0.0, 0.0, 0.0);
        Vec3 vq = Vec3(qrot[1], qrot[2], qrot[3]);
        double q0 = qrot[0];
        for (int qidx = 0; qidx < 4; qidx++ ){
            if (rotate){
                dq0_1[qidx] = quaternionRotate(qrot, dq0_1[qidx]);
            }
                force += -dq0_1[qidx] * deriv_const[qidx]/numParticles;
        }

        forces_buffer[particles[ia]] = mm_double4(force[0], force[1], force[2], 0.0);
        //forces_buffer[particles[ia]] = mm_double4(0.0002, 0.0002, 0.0002, 0.0);

        } // cacl grad
    }

std::vector<Vec3> rotateCoordinates(std::vector<double> qr, const std::vector<Vec3> pos){
    std::vector<Vec3> rot_pos;
    unsigned ntot = pos.size();
    //rot_pos.resize(ntot, Vec3(0,0,0));
    for (unsigned i=0; i<ntot; i++)
            rot_pos.push_back((quaternionRotate(qr, pos[i])));

    return rot_pos;
}

template <class REAL>
void calc_correlation_matrix(bool usecenter, bool rotate, const std::vector<OpenMM::Vec3> refpos, const std::vector<OpenMM::Vec3> pos, vector<int> particles,
                            std::vector<REAL>& C, Vec3 poscenter=Vec3(0.0, 0.0, 0.0), vector<double> q={0.0, 0.0, 0.0, 0.0}){
    vector<Vec3> pos1, pos2;
    for (int i : particles){
        pos1.push_back(refpos[i]);
        pos2.push_back(pos[i]);
    }
    Vec3 center = Vec3(0.0, 0.0, 0.0);
    if (!usecenter)
    {
    for (Vec3 p : pos2)
        center += p;

    center /= pos2.size();

    }
    else
        center = poscenter;


    for (Vec3& p : pos2)
        p -= center;

    if (rotate)
        pos2 = rotateCoordinates(q, pos2);

    double C1[3][3];
    memset(C1, 0, sizeof(C1));
    for (int k = 0; k < pos1.size(); k++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                C1[i][j] += pos1[k][i]*pos2[k][j];


    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        {
        C[3*i+j] = C1[i][j];
        }
}
void CommonCalcEuleranglesForceKernel::initialize(const System& system, const EuleranglesForce& force) {
    ContextSelector selector(cc);
    bool useDouble = cc.getUseDoublePrecision();
    int elementSize = (useDouble ? sizeof(double) : sizeof(float));
    int numParticles = force.getParticles().size();
    angle = force.get_Angle();
    if (numParticles == 0)
        numParticles = system.getNumParticles();

    if (numParticles < 50)
        enable_cpu = true;

    int fit_numParticles = force.getFittingParticles().size();
    if (fit_numParticles != 0) {
        enable_fitting = true;
        fit_referencePos.initialize(cc, system.getNumParticles(), 4*elementSize, "referencePos");
        fit_particles.initialize<int>(cc, fit_numParticles, "particles");
        fit_buffer.initialize(cc, 12, elementSize, "buffer");
    }

    referencePos.initialize(cc, system.getNumParticles(), 4*elementSize, "referencePos");
    forces.initialize(cc, system.getNumParticles(), 4*elementSize, "forces");
    particles.initialize<int>(cc, numParticles, "particles");
    buffer.initialize(cc, 12, elementSize, "buffer");
    eigval.initialize(cc, 4, elementSize, "eigval");
    eigvec.initialize(cc, 4, 4*elementSize, "eigvec");
    poscenter.initialize(cc, 3, elementSize, "poscenter");
    qrot.initialize(cc, 4, elementSize, "qrot");
    qrot_deriv.initialize(cc, 4, elementSize, "qrot_dev");
    anglederiv.initialize(cc, 4, elementSize, "anglederiv");


    recordParameters(force);
    info = new CommonEuleranglesForceInfo(force);
    cc.addForce(info);

    // Create the kernels.
    // importnat variable
    blockSize = min(256, cc.getMaxThreadBlockSize());
    map<string, string> defines;
    defines["THREAD_BLOCK_SIZE"] = cc.intToString(blockSize);
    ComputeProgram program = cc.compileProgram(CommonEuleranglesKernelSources::euleranglesForce, defines);
    kernel1 = program->createKernel("computeEuleranglesPart1");
    kernel2 = program->createKernel("computeEuleranglesForces");
    kernel1->addArg();
    kernel1->addArg();
    kernel1->addArg();
    kernel1->addArg(cc.getPosq());
    kernel1->addArg(referencePos);
    kernel1->addArg(particles);
    kernel1->addArg(poscenter);
    kernel1->addArg(qrot);
    kernel1->addArg(buffer);
    kernel2->addArg();
    kernel2->addArg();
    kernel2->addArg(cc.getPaddedNumAtoms());
    kernel2->addArg(cc.getPosq());
    kernel2->addArg(referencePos);
    kernel2->addArg(particles);
    kernel2->addArg(eigval);
    kernel2->addArg(eigvec);
    kernel2->addArg(anglederiv);
    kernel2->addArg(qrot_deriv);
    kernel2->addArg(cc.getLongForceBuffer());
    if (enable_cpu) {
    kernel4 = program->createKernel("addForces");
    kernel4->addArg();
    kernel4->addArg(cc.getPaddedNumAtoms());
    kernel4->addArg(particles);
    kernel4->addArg(forces);
    kernel4->addArg(cc.getLongForceBuffer());
    }

    // for fitting group
    if (enable_fitting != 0) {
        //fit_poscenter.initialize(cc, 3, elementSize, "poscenter");
        kernel3 = program->createKernel("computeEuleranglesPart1");
        kernel3->addArg();
        kernel3->addArg();
        kernel3->addArg();
        kernel3->addArg(cc.getPosq());
        kernel3->addArg(fit_referencePos);
        kernel3->addArg(fit_particles);
        kernel3->addArg(poscenter);
        kernel3->addArg(qrot);
        kernel3->addArg(fit_buffer);
    }

}

void CommonCalcEuleranglesForceKernel::recordParameters(const EuleranglesForce& force) {
    // Record the parameters and center the reference positions.

    particleVec = force.getParticles();
    if (particleVec.size() == 0)
        for (int i = 0; i < cc.getNumAtoms(); i++)
            particleVec.push_back(i);
    centeredPositions = force.getReferencePositions();
    Vec3 center;
    for (int i : particleVec)
        center += centeredPositions[i];

    center /= particleVec.size();
    for (Vec3& p : centeredPositions)
        p -= center;
    // Upload them to the device.

    particles.upload(particleVec);
    vector<mm_double4> pos;
    for (Vec3 p : centeredPositions)
        pos.push_back(mm_double4(p[0], p[1], p[2], 0));
    referencePos.upload(pos, true);

   // fit_particles
   if (enable_fitting){
        fit_particleVec = force.getFittingParticles();
        fit_centeredPositions = force.getReferencePositions();
        center;
        for (int i : fit_particleVec)
            center += fit_centeredPositions[i];
        center /= fit_particleVec.size();
        for (Vec3& p : fit_centeredPositions)
            p -= center;

        // Upload them to the device.
        fit_particles.upload(fit_particleVec);
        vector<mm_double4> pos;
        for (Vec3 p : fit_centeredPositions)
            pos.push_back(mm_double4(p[0], p[1], p[2], 0));
        fit_referencePos.upload(pos, true);

   }
}

double CommonCalcEuleranglesForceKernel::execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) {
    ContextSelector selector(cc);
    if (cc.getUseDoublePrecision())
        return executeImpl<double>(context);
    return executeImpl<float>(context);
}

template <class REAL>
double CommonCalcEuleranglesForceKernel::executeImpl(OpenMM::ContextImpl& context) {
    // Execute the first kernel.
    double energy;
    if (!enable_fitting) {
        if (!enable_cpu){
            // run on CUDA
            int numParticles = particles.getSize();
            kernel1->setArg(0, numParticles);
            kernel1->setArg(1, false);
            kernel1->setArg(2, false);
            kernel1->execute(blockSize, blockSize);
            // Download the Correlation matrix, build the S matrix, and find the maximum eigenvalue
            // and eigenvector.
            vector<REAL> C;
            buffer.download(C);

            // JAMA::Eigenvalue may run into an infinite loop if we have any NaN
            for (int i = 0; i < 9; i++) {
                if (C[i] != C[i])
                    throw OpenMMException("NaN encountered during Eulerangles force calculation");
            }
            vector<REAL> eigval_buffer, anglederiv_buffer(4);
            vector<mm_double4> eigvec_buffer;
            vector<double> q;
            calculateQRotation(C, eigvec_buffer, eigval_buffer, q);
            calculateDeriv(q, angle, anglederiv_buffer, energy);
            // upload data to calculate forces
            eigval.upload(eigval_buffer);
            eigvec.upload(eigvec_buffer, true);
            anglederiv.upload(anglederiv_buffer);
            kernel2->setArg(0, numParticles);
            kernel2->setArg(1, false);
            kernel2->execute(numParticles);
        }
        else{
            int numParticles = particles.getSize();
            vector<Vec3> MDPositions;
            vector<REAL> C(9);
            context.getPositions(MDPositions);
            calc_correlation_matrix(false, false, centeredPositions, MDPositions, particleVec, C);

            // JAMA::Eigenvalue may run into an infinite loop if we have any NaN
            for (int i = 0; i < 9; i++) {
                if (C[i] != C[i])
                    throw OpenMMException("NaN encountered during Eulerangles force calculation");
            }
            vector<mm_double4> force_buffer;
            force_buffer.resize(MDPositions.size(), mm_double4(0.0, 0.0, 0.0, 0.0));
            calcForces(C, centeredPositions, particleVec, angle, force_buffer, energy);
            forces.upload(force_buffer, true);
            kernel4->setArg(0, numParticles);
            kernel4->execute(1);
        }
    }

    else{
        if (!enable_cpu){
            //center the current positions using the center of fitting group atoms
            int fit_numParticles = fit_particles.getSize();
            kernel3->setArg(0, fit_numParticles);
            kernel3->setArg(1, false);
            kernel3->setArg(2, false);
            kernel3->execute(blockSize, blockSize);
            // Download the Correlation matrix, build the S matrix, and find the maximum eigenvalue
            // and eigenvector.
            vector<REAL> fit_C;
            fit_buffer.download(fit_C);
            // JAMA::Eigenvalue may run into an infinite loop if we have any NaN
            for (int i = 0; i < 9; i++) {
                if (fit_C[i] != fit_C[i])
                    throw OpenMMException("NaN encountered during Eulerangles force calculation");
            }
            // compute optimal rotation
            vector<REAL> fit_center = {fit_C[9], fit_C[10], fit_C[11]};
            vector<REAL> fit_eigval_buffer, anglederiv_buffer(4);
            vector<double> fit_q;
            vector<mm_double4> fit_eigvec_buffer;
            calculateQRotation(fit_C, fit_eigvec_buffer, fit_eigval_buffer, fit_q);
            //calculateDeriv(fit_q, angle, anglederiv_buffer, energy);

            // center, apply fit rotation and calculate optimal rotation between particles and its reference
            // center particles using the cog of fitting group atoms

            vector<REAL> qrot_buffer = {static_cast<REAL>(fit_q[0]), static_cast<REAL>(fit_q[1]),
                                        static_cast<REAL>(fit_q[2]), static_cast<REAL>(fit_q[3])};
            int numParticles = particles.getSize();
            poscenter.upload(fit_center);
            qrot.upload(qrot_buffer);
            kernel1->setArg(0, numParticles);
            kernel1->setArg(1, true); // uses the given center for recentering
            kernel1->setArg(2, true); // rotates using the fit_q rotation
            kernel1->execute(blockSize, blockSize);
            // Download the Correlation matrix, build the S matrix, and find the maximum eigenvalue
            // and eigenvector.
            vector<REAL> C;
            buffer.download(C);

            // JAMA::Eigenvalue may run into an infinite loop if we have any NaN
            for (int i = 0; i < 9; i++) {
                if (C[i] != C[i])
                    throw OpenMMException("NaN encountered during Eulerangles force calculation");
            }
            vector<REAL> eigval_buffer;
            vector<mm_double4> eigvec_buffer;
            vector<double> q;
            calculateQRotation(C, eigvec_buffer, eigval_buffer, q);
            calculateDeriv(q, angle, anglederiv_buffer, energy);

            vector<REAL> inv_qrot_buffer = {static_cast<REAL>(fit_q[0]), -static_cast<REAL>(fit_q[1]),
                                        -static_cast<REAL>(fit_q[2]), -static_cast<REAL>(fit_q[3])};
            // // compute forces forces
            eigval.upload(eigval_buffer);
            eigvec.upload(eigvec_buffer, true);
            qrot_deriv.upload(inv_qrot_buffer);
            anglederiv.upload(anglederiv_buffer);
            kernel2->setArg(0, numParticles);
            kernel2->setArg(1, true); // apply fitting rotation
            kernel2->execute(numParticles);
        }
        else{
            //center the current positions using the center of fitting group atoms
            int fit_numParticles = fit_particles.getSize();
            // kernel3->setArg(0, fit_numParticles);
            // kernel3->setArg(1, false);
            // kernel3->setArg(2, false);
            // kernel3->execute(blockSize, blockSize);
            // // Download the Correlation matrix, build the S matrix, and find the maximum eigenvalue
            // // and eigenvector.
            // vector<REAL> fit_C;
            // fit_buffer.download(fit_C);
            vector<Vec3> MDPositions;
            vector<REAL> fit_C(9);
            context.getPositions(MDPositions);
            calc_correlation_matrix(false, false, fit_centeredPositions, MDPositions, fit_particleVec, fit_C);
            // JAMA::Eigenvalue may run into an infinite loop if we have any NaN
            for (int i = 0; i < 9; i++) {
                if (fit_C[i] != fit_C[i])
                    throw OpenMMException("NaN encountered during Eulerangles force calculation");
            }
            // compute optimal rotation
            // Download the Correlation matrix, build the S matrix, and find the maximum eigenvalue
            // and eigenvector.
            vector<REAL> eigval_buffer;
            vector<mm_double4> eigvec_buffer;
            vector<double> fit_q;
            calculateQRotation(fit_C, eigvec_buffer, eigval_buffer, fit_q);

            vector<REAL> C(9);
            Vec3 fit_cog = calculateCOG(MDPositions, fit_particleVec);
            context.getPositions(MDPositions);
            calc_correlation_matrix(true, true, centeredPositions, MDPositions, particleVec, C, fit_cog, fit_q);
            // JAMA::Eigenvalue may run into an infinite loop if we have any NaN
            for (int i = 0; i < 9; i++) {
                if (C[i] != C[i])
                    throw OpenMMException("NaN encountered during Eulerangles force calculation");
            }
            // vector<double> q;
            // calculateQRotation(C, eigvec_buffer, eigval_buffer, q);
            // double radian_to_degree = 180 / 3.1415926;
            // double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3];
            // double x = 2 * (q1 * q3 - q4 * q2);
            // energy = radian_to_degree * asin(x);
            vector<mm_double4> force_buffer;
            force_buffer.resize(MDPositions.size(), mm_double4(0.0, 0.0, 0.0, 0.0));

            vector<double> inv_fit_q = {fit_q[0], -fit_q[1], -fit_q[2], -fit_q[3]};
            calcForces(C, centeredPositions, particleVec, angle, force_buffer, energy, true, inv_fit_q);
            int numParticles = particles.getSize();
            forces.upload(force_buffer, true);
            kernel4->setArg(0, numParticles);
            kernel4->execute(1);

        }

    }

    return energy;
}

void CommonCalcEuleranglesForceKernel::copyParametersToContext(OpenMM::ContextImpl& context, const EuleranglesForce& force) {
    ContextSelector selector(cc);
    if (referencePos.getSize() != force.getReferencePositions().size())
        throw OpenMMException("updateParametersInContext: The number of reference positions has changed");
    int numParticles = force.getParticles().size();
    if (numParticles == 0)
        numParticles = context.getSystem().getNumParticles();
    if (numParticles != particles.getSize())
        particles.resize(numParticles);
    recordParameters(force);

    // Mark that the current reordering may be invalid.

   info->updateParticles();
   cc.invalidateMolecules(info);
}
