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

using namespace EuleranglesPlugin;
using namespace OpenMM;
using namespace std;

template <class REAL>
void calculateQRotation(std::vector<REAL> C, vector<mm_double4>& eigvec_buffer, vector<REAL>& eigval_buffer, vector<double>& q){
    Array2D<double> S(4, 4);
    S[0][0] = -C[0*3+0] - C[1*3+1] - C[2*3+2];
    S[1][0] = -C[1*3+2] + C[2*3+1];
    S[2][0] = -C[2*3+0] + C[0*3+2];
    S[3][0] = -C[0*3+1] + C[1*3+0];
    S[0][1] = -C[1*3+2] + C[2*3+1];
    S[1][1] = -C[0*3+0] + C[1*3+1] + C[2*3+2];
    S[2][1] = -C[0*3+1] - C[1*3+0];
    S[3][1] = -C[0*3+2] - C[2*3+0];
    S[0][2] = -C[2*3+0] + C[0*3+2];
    S[1][2] = -C[0*3+1] - C[1*3+0];
    S[2][2] =  C[0*3+0] - C[1*3+1] + C[2*3+2];
    S[3][2] = -C[1*3+2] - C[2*3+1];
    S[0][3] = -C[0*3+1] + C[1*3+0];
    S[1][3] = -C[0*3+2] - C[2*3+0];
    S[2][3] = -C[1*3+2] - C[2*3+1];
    S[3][3] =  C[0*3+0] + C[1*3+1] - C[2*3+2];

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
    
    // inverse eigen vectrs 
    // inverse for eulerangles inv(q) = (q0, -q1, -q, -q3)
    for (int i=0;i<4;i++) {
        for (int j=1;j<4;j++) 
            S_eigvec[i][j] = -S_eigvec[i][j];
    }
    // set the q
    for (int i=0; i<4; i++)
        q[i] = S_eigvec[0][i]; 
    eigval_buffer = {static_cast<REAL>(S_eigval[0]), static_cast<REAL>(S_eigval[1]), 
                                static_cast<REAL>(S_eigval[2]), static_cast<REAL>(S_eigval[3])}; 
    
    for (int i=0;i<4;i++) 
        eigvec_buffer.push_back(mm_double4(S_eigvec[i][0], S_eigvec[i][1], S_eigvec[i][2], S_eigvec[i][3]));
}
double asinDerivative(double x) {
    return 1 / sqrt(1 - x * x);
}

template <class REAL>
void calculateDeriv(std::vector<double> q, std::string angle, vector<REAL>& anglederiv_buffer, double& energy){
    double radian_to_degree = 180 / 3.1415926;
    if (angle == "Euler") {
        double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3];
        double x = 2 * (q1 * q3 - q4 * q2);
        energy = radian_to_degree * asin(x);
        double deriv = 2 * radian_to_degree * asinDerivative(x);
        anglederiv_buffer[0] = static_cast<REAL>(deriv * q3);
        anglederiv_buffer[1] = static_cast<REAL>(-deriv * q4);
        anglederiv_buffer[2] = static_cast<REAL>(deriv * q1);
        anglederiv_buffer[3] = static_cast<REAL>(-deriv * q2);
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
    referencePos.initialize(cc, system.getNumParticles(), 4*elementSize, "referencePos");
    particles.initialize<int>(cc, numParticles, "particles");
    buffer.initialize(cc, 9, elementSize, "buffer");
    eigval.initialize(cc, 4, elementSize, "eigval");
    eigvec.initialize(cc, 4, 4*elementSize, "eigvec");
    poscenter.initialize(cc, 3, elementSize, "poscenter");
    qrot.initialize(cc, 4, elementSize, "qrot");
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
    kernel2->addArg(cc.getPaddedNumAtoms());
    kernel2->addArg(cc.getPosq());
    kernel2->addArg(referencePos);
    kernel2->addArg(particles);
    kernel2->addArg(eigval);
    kernel2->addArg(eigvec);
    kernel2->addArg(anglederiv);
    kernel2->addArg(cc.getLongForceBuffer());
    // for fitting group 
    int fit_numParticles = force.getFittingParticles().size();
    if (fit_numParticles != 0) {
        enable_fitting = true;
        fit_referencePos.initialize(cc, system.getNumParticles(), 4*elementSize, "referencePos");
        fit_particles.initialize<int>(cc, fit_numParticles, "particles");
        fit_buffer.initialize(cc, 9, elementSize, "buffer");
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
    
    vector<int> particleVec = force.getParticles();
    if (particleVec.size() == 0)
        for (int i = 0; i < cc.getNumAtoms(); i++)
            particleVec.push_back(i);
    vector<Vec3> centeredPositions = force.getReferencePositions();
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
        vector<int> particleVec = force.getFittingParticles();
        vector<Vec3> centeredPositions = force.getReferencePositions();
        Vec3 center;
        for (int i : particleVec)
            center += centeredPositions[i];
        center /= particleVec.size();
        for (Vec3& p : centeredPositions)
            p -= center;

        // Upload them to the device.
        fit_particles.upload(particleVec);
        vector<mm_double4> pos;
        for (Vec3 p : centeredPositions)
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
        int numParticles = particles.getSize();
        kernel1->setArg(0, numParticles);
        kernel1->setArg(1, true);
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
        vector<double> q(4);
        calculateQRotation(C, eigvec_buffer, eigval_buffer, q);
    
        calculateDeriv(q, angle, anglederiv_buffer, energy);
   
        // upload data to calculate forces 
        eigval.upload(eigval_buffer);
        eigvec.upload(eigvec_buffer, true);
        anglederiv.upload(anglederiv_buffer);
        kernel2->setArg(0, numParticles);
        kernel2->execute(numParticles);
    }
    else{
        // center the current positions using the center of fitting group atoms
        int fit_numParticles = fit_particles.getSize();
        kernel3->setArg(0, fit_numParticles);
        kernel3->setArg(1, true);
        kernel3->setArg(2, false);
        kernel3->execute(blockSize, blockSize);
        vector<REAL> fit_C; 
        fit_buffer.download(fit_C);
        // compute optimal rotation 
        vector<REAL> fit_eigval_buffer;
        vector<mm_double4> fit_eigvec_buffer;
        vector<double> fit_q(4);
        calculateQRotation(fit_C, fit_eigvec_buffer, fit_eigval_buffer, fit_q);  
        vector<mm_double4> eigvec_buffer;
        vector<REAL> eigval_buffer, anglederiv_buffer(4);
        calculateDeriv(fit_q, angle, anglederiv_buffer, energy);
        cout<< energy << "\n";
        // center, apply fit rotation and calculate optimal rotation between particles and its reference
        // center particles using the cog of fitting group atoms
        // vector<REAL> fit_center = {static_cast<REAL>(fit_C[10]), static_cast<REAL>(fit_C[11]), static_cast<REAL>(fit_C[12])}; 
        // vector<REAL> qrot_buffer = {static_cast<REAL>(fit_q[0]), static_cast<REAL>(fit_q[1]), 
        //                      static_cast<REAL>(fit_q[2]), static_cast<REAL>(fit_q[3])};
        // int numParticles = particles.getSize();
        // poscenter.upload(fit_center);
        // qrot.upload(qrot_buffer);
        // kernel1->setArg(0, numParticles);
        // kernel1->setArg(1, true);
        // kernel1->setArg(2, false);
        // kernel1->execute(blockSize, blockSize);        
        // vector<REAL> C; 
        // fit_buffer.download(C);   
        
        // // compute particles rotation q
        // vector<REAL> eigval_buffer, anglederiv_buffer(4);
        // vector<mm_double4> eigvec_buffer;
        // vector<double> q(4);
        // calculateQRotation(C, eigvec_buffer, eigval_buffer, q);
        // cout << q[0] << "\t" << q[1] << "\t" << q[2] << "\n";
        // calculateDeriv(q, angle, anglederiv_buffer, energy);
        // cout<< energy << "\n";
        // apply forces 
             
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
