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

#include "CommonPolaranglesKernels.h"
#include "CommonPolaranglesKernelSources.h"
#include "openmm/common/BondedUtilities.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include "jama_eig.h"
#include <set>

using namespace PolaranglesPlugin;
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
    // inverse for polarangles inv(q) = (q0, -q1, -q, -q3)
    for (int i=0;i<4;i++) {
        for (int j=1;j<4;j++) 
            S_eigvec[i][j] = -S_eigvec[i][j];
    }
    // set q
    for (int i=0; i<4; i++)
        q.push_back(S_eigvec[0][i]); 
    eigval_buffer = {static_cast<REAL>(S_eigval[0]), static_cast<REAL>(S_eigval[1]), 
                                static_cast<REAL>(S_eigval[2]), static_cast<REAL>(S_eigval[3])};                         
    
    for (int i=0;i<4;i++) 
        eigvec_buffer.push_back(mm_double4(S_eigvec[i][0], S_eigvec[i][1], S_eigvec[i][2], S_eigvec[i][3]));
}

double acosDerivative(double x) {
  // Check if the input is within the valid range of -1 to 1
  if (x < -1.0 || x > 1.0) {
    // Handle invalid input, return NaN or throw an exception
    // depending on your requirements.
    // For simplicity, let's return NaN in this example.
    throw OpenMMException("updateParametersInContext: The angle values are NaN");
    return NAN;
  }
  
  // Calculate the derivative of acos(x)
  return -1.0 / sqrt(1.0 - x*x);
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
void calculateDeriv(Vec3 unit_vec, std::string angle, vector<REAL>& deriv_const, double& energy){
    double radian_to_degree = 180 / 3.1415926;
    double i1 = unit_vec[0], i2 = unit_vec[1],  i3 = unit_vec[2];
    if (angle == "Theta") {
        energy = radian_to_degree * acos(-i2);
        deriv_const[0] = static_cast<REAL>(0.0);
        deriv_const[1] = static_cast<REAL>(-acosDerivative(-i2)); 
        deriv_const[2] = static_cast<REAL>(0.0);
    } 
    else if (angle == "Phi"){
        energy = radian_to_degree * atan2(i3, i1);
        double deriv_x, deriv_y;
        atan2Derivatives(i3, i1, deriv_x, deriv_y);
        deriv_const[0] = static_cast<REAL>(radian_to_degree * deriv_y);
        deriv_const[1] = static_cast<REAL>(0.0);
        deriv_const[2] = static_cast<REAL>(radian_to_degree * deriv_x);
    }
    else
        throw OpenMMException("updateParametersInContext: The angle type is not correct");   
}

// From OpenMM referenceForce
double modulo(const Vec3& deltaR) {
  
    return sqrt(deltaR[0]*deltaR[0] + deltaR[1]*deltaR[1] + deltaR[2]*deltaR[2]);
}

Vec3 getDeltaR(const Vec3& atomCoordinatesI, const Vec3& atomCoordinatesJ) {
    return atomCoordinatesJ - atomCoordinatesI;
}

void CommonCalcPolaranglesForceKernel::initialize(const System& system, const PolaranglesForce& force) {
    ContextSelector selector(cc);
    bool useDouble = cc.getUseDoublePrecision();
    int elementSize = (useDouble ? sizeof(double) : sizeof(float));
    int numParticles = force.getParticles().size();
    angle = force.get_Angle();
    if (numParticles == 0)
        numParticles = system.getNumParticles();
        
    int fit_numParticles = force.getFittingParticles().size();
    fit_referencePos.initialize(cc, system.getNumParticles(), 4*elementSize, "referencePos");
    fit_particles.initialize<int>(cc, fit_numParticles, "particles");
    fit_buffer.initialize(cc, 12, elementSize, "buffer");
        
    referencePos.initialize(cc, system.getNumParticles(), 4*elementSize, "referencePos");
    particles.initialize<int>(cc, numParticles, "particles");
    buffer.initialize(cc, 12, elementSize, "buffer");
    derive_matrix.initialize(cc, 3, 4*elementSize, "derive_matrix");
    poscenter.initialize(cc, 3, elementSize, "poscenter");
    qrot.initialize(cc, 4, elementSize, "qrot");
    qrot_deriv.initialize(cc, 4, elementSize, "qrot_dev");
    anglederiv.initialize(cc, 3, elementSize, "anglederiv");
    center_buffer.initialize(cc, 3, elementSize, "centr_buffer");
    
    
    recordParameters(force);
    info = new CommonPolaranglesForceInfo(force);
    cc.addForce(info);
   
    // Create the kernels.
    // importnat variable
    blockSize = min(256, cc.getMaxThreadBlockSize());
    map<string, string> defines;
    defines["THREAD_BLOCK_SIZE"] = cc.intToString(blockSize);
    ComputeProgram program = cc.compileProgram(CommonPolaranglesKernelSources::polaranglesForce, defines);
    kernel1 = program->createKernel("computePolaranglesPart1");
    kernel2 = program->createKernel("computePolaranglesForces");
    kernel3 = program->createKernel("computecCOG");
    kernel4 = program->createKernel("computecCOG");
    // compute correlation matrix 
    kernel1->addArg();
    kernel1->addArg();
    kernel1->addArg();
    kernel1->addArg(cc.getPosq());
    kernel1->addArg(fit_referencePos);
    kernel1->addArg(fit_particles);
    kernel1->addArg(poscenter);
    kernel1->addArg(qrot);
    kernel1->addArg(fit_buffer); 
    
    // compute COG for the particles
    kernel3->addArg();
    kernel3->addArg(cc.getPosq());
    kernel3->addArg(particles);
    kernel3->addArg(poscenter);
    kernel3->addArg(qrot);
    kernel3->addArg(center_buffer);
    
    kernel4->addArg();
    kernel4->addArg(cc.getPosq());
    kernel4->addArg(fit_particles);
    kernel4->addArg(poscenter);
    kernel4->addArg(qrot);
    kernel4->addArg(center_buffer);
    
    // calculate forces 
    kernel2->addArg();
    kernel2->addArg(cc.getPaddedNumAtoms());
    kernel2->addArg(particles);
    kernel2->addArg(derive_matrix);
    kernel2->addArg(anglederiv);
    kernel2->addArg(qrot_deriv);
    kernel2->addArg(cc.getLongForceBuffer());
 }

void CommonCalcPolaranglesForceKernel::recordParameters(const PolaranglesForce& force) {
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

    particleVec = force.getFittingParticles();
    centeredPositions = force.getReferencePositions();
    center;
    for (int i : particleVec)
        center += centeredPositions[i];
    center /= particleVec.size();
    for (Vec3& p : centeredPositions)
        p -= center;

    // Upload them to the device.
    vector<mm_double4> fit_pos;
    fit_particles.upload(particleVec);
    for (Vec3 p : centeredPositions)
        fit_pos.push_back(mm_double4(p[0], p[1], p[2], 0));
    fit_referencePos.upload(fit_pos, true);
    

}

double CommonCalcPolaranglesForceKernel::execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) {
    ContextSelector selector(cc);
    if (cc.getUseDoublePrecision())
        return executeImpl<double>(context);
    return executeImpl<float>(context);
}

template <class REAL>
double CommonCalcPolaranglesForceKernel::executeImpl(OpenMM::ContextImpl& context) {
    // Execute the first kernel.
    double energy;

    //center the current positions using the center of fitting group atoms
    int fit_numParticles = fit_particles.getSize();
    kernel1->setArg(0, fit_numParticles);
    kernel1->setArg(1, false); 
    kernel1->setArg(2, false);
    kernel1->execute(blockSize, blockSize);
    // Download the Correlation matrix, build the S matrix, and find the maximum eigenvalue
    // and eigenvector.
    vector<REAL> fit_C; 
    fit_buffer.download(fit_C);
    // JAMA::Eigenvalue may run into an infinite loop if we have any NaN
    for (int i = 0; i < 9; i++) {
        if (fit_C[i] != fit_C[i])
            throw OpenMMException("NaN encountered during Polarangles force calculation");
    }
    // compute optimal rotation 
    //vector<REAL> fit_center = {static_cast<REAL>(fit_C[10]), static_cast<REAL>(fit_C[11]), static_cast<REAL>(fit_C[12])}; 
    vector<REAL> fit_center = {fit_C[9], fit_C[10], fit_C[11]};
    //cout<< fit_center[0] << "\t" << fit_center[1]<< "\t" << fit_center[2] << "\n"; 
    vector<REAL> fit_eigval_buffer;
    vector<mm_double4> fit_eigvec_buffer;
    vector<double> fit_q;
    calculateQRotation(fit_C, fit_eigvec_buffer, fit_eigval_buffer, fit_q);     
    vector<REAL> qrot_buffer = {static_cast<REAL>(fit_q[0]), static_cast<REAL>(fit_q[1]), 
                                static_cast<REAL>(fit_q[2]), static_cast<REAL>(fit_q[3])};
    
    // compute COG of the rotated particles pos
    int numParticles = particles.getSize(); 
    poscenter.upload(fit_center);
    qrot.upload(qrot_buffer);
    kernel3->setArg(0, numParticles);
    kernel3->execute(blockSize, blockSize);
    vector<REAL> rotpos_cog_buf;
    center_buffer.download(rotpos_cog_buf);
    
    poscenter.upload(fit_center);
    qrot.upload(qrot_buffer);
    kernel4->setArg(0, fit_numParticles);
    kernel4->execute(blockSize, blockSize);
    vector<REAL> fit_rotpos_cog_buf;
    center_buffer.download(fit_rotpos_cog_buf);
    
    Vec3 rotpos_cog, fit_rotpos_cog; 
    rotpos_cog = Vec3(rotpos_cog_buf[0], rotpos_cog_buf[1], rotpos_cog_buf[2]);
    fit_rotpos_cog = Vec3(fit_rotpos_cog_buf[0], fit_rotpos_cog_buf[1], fit_rotpos_cog_buf[2]);
    Vec3 distance = getDeltaR(fit_rotpos_cog, rotpos_cog);
    double norm = modulo(distance);
    Vec3 unit_vec = distance / norm;
 

    vector<REAL> anglederiv_buffer(3);
    calculateDeriv(unit_vec, angle, anglederiv_buffer, energy);
    
    std::vector <Vec3> deriv_matrix_vec = {Vec3(1.0, 0.0, 0.0) - unit_vec[0] * unit_vec,
                                       Vec3(0.0, 1.0, 0.0) - unit_vec[1] * unit_vec,
                                       Vec3(0.0, 0.0, 1.0) - unit_vec[2] * unit_vec};
    vector<mm_double4> deriv_matrix_vec_buffer;
    for (Vec3 p : deriv_matrix_vec)
        deriv_matrix_vec_buffer.push_back(mm_double4(p[0], p[1], p[2], 0));
     
    vector<REAL> inv_qrot_buffer = {static_cast<REAL>(fit_q[0]), -static_cast<REAL>(fit_q[1]), 
                                    -static_cast<REAL>(fit_q[2]), -static_cast<REAL>(fit_q[3])};
    // // // compute forces forces 
    derive_matrix.upload(deriv_matrix_vec_buffer, true);
    anglederiv.upload(anglederiv_buffer);
    qrot_deriv.upload(inv_qrot_buffer);
    kernel2->setArg(0, numParticles);
    kernel2->execute(numParticles);              

    return energy;
}

void CommonCalcPolaranglesForceKernel::copyParametersToContext(OpenMM::ContextImpl& context, const PolaranglesForce& force) {
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
