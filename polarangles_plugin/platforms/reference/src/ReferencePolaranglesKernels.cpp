/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
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

#include "ReferencePolaranglesKernels.h"
#include "PolaranglesForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/Vec3.h"
#include "jama_eig.h"
#include "Quaternion.h"

#include <cmath>
using namespace PolaranglesPlugin;
using namespace OpenMM;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
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

// From OpenMM referenceForce
double modulo(const Vec3& deltaR) {
  
   deltaR[R2Index] = DOT3(deltaR, deltaR);
   return sqrt(deltaR[R2Index]);
}

Vec3 getDeltaR(const Vec3& atomCoordinatesI, const Vec3& atomCoordinatesJ) {
    return atomCoordinatesJ-atomCoordinatesI;
}

ReferenceCalcPolaranglesForceKernel::~ReferenceCalcPolaranglesForceKernel() {
}


Vec3 calculateCOG(const std::vector<Vec3> pos)
{
    Vec3 cog = Vec3(0,0,0);
    for (unsigned i=0; i<pos.size(); i++)
            cog += pos[i];
    cog /= pos.size();
    return cog;
}

std::vector<Vec3> translateCoordinates(const std::vector<Vec3> pos, const Vec3 t)
{
    std::vector<Vec3> translated_pos;
    for (unsigned i=0; i<pos.size(); i++) {
        translated_pos.push_back(pos[i] - t);
    }
 return translated_pos;
}

std::vector<Vec3> shiftbyCOG(const std::vector<Vec3> pos)
{
   Vec3 t = calculateCOG(pos);
   return translateCoordinates(pos, t);
}

void calculateDeriv(std::vector<double> q, std::string angle, std::vector<double>& deriv_const, double& energy){
    double radian_to_degree = 180 / 3.1415926;
    double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3];
    if (angle == "Polar") {
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
         throw OpenMMException("updateParametersInContext: The angle type is not defined");
    
}

void ReferenceCalcPolaranglesForceKernel::initialize(const System& system, const PolaranglesForce& force) {
    particles = force.getParticles();
    fitting_particles = force.getFittingParticles();
    angle = force.get_Angle();
    if (particles.size() == 0)
        for (int i = 0; i < system.getNumParticles(); i++)
            particles.push_back(i);

    referencePos = force.getReferencePositions();
}

double ReferenceCalcPolaranglesForceKernel::calculateIxn(vector<OpenMM::Vec3>& atomCoordinates, vector<OpenMM::Vec3>& forces) const {
    // Compute the Quaternion and its gradient using the algorithm described in Coutsias et al,
    // "Using Quaternion to calculate Quaternion" (doi: 10.1002/jcc.20110).  First subtract
    // the centroid from the atom positions.  The reference positions have already been centered.
    std::vector<double> normquat = {1.0, 0.0, 0.0, 0.0}; 
    Quaternion qrot, fit_qrot;;

        
    std::vector<Vec3> refpos, pos, fit_refpos, fit_pos, centered_refpos, centered_pos, fit_centered_refpos, fit_centered_pos; 
    for (int i : particles ){
        refpos.push_back(referencePos[i]);
        pos.push_back(atomCoordinates[i]);
    } 
    
    for (int i : fitting_particles){
        fit_refpos.push_back(referencePos[i]);
        fit_pos.push_back(atomCoordinates[i]);
    }
    // center reference pos
    centered_refpos = shiftbyCOG(refpos);
    fit_centered_refpos = shiftbyCOG(fit_refpos);
    
    // center current posittions 
    Vec3 fit_cog = calculateCOG(fit_pos);
    centered_pos = translateCoordinates(pos, fit_cog);
    fit_centered_refpos = translateCoordinates(fit_pos, fit_cog);

    // You need to request gradients first
    fit_qrot.request_group1_gradients(fit_pos.size());
    fit_qrot.calc_optimal_rotation(fit_centered_pos, fit_centered_refpos, normquat);

    // apply fitting rotation
    std::vector<Vec3> fit_rot_pos, rot_pos;
    fit_rot_pos = fit_qrot.rotateCoordinates(fit_qrot.q, fit_centered_pos);
    rot_pos = fit_qrot.rotateCoordinates(fit_qrot.q, centered_pos);

    Vec3 fit_rot_pos_com, rot_pos_com;
    fit_rot_pos_com = calculateCOG(fit_rot_pos);
    rot_pos_com = calculateCOG(rot_pos);
        
    Vec3 distance = getDeltaR(fit_rot_pos_com, rot_pos_com);
    doubel norm = modulo(distance);
    Vector unit_vec = distance / norm;
    
    // calcualte derivatives 
    // double energy;
    // vector<double> deriv_const(4);
    // calculateDeriv(qrot.q, angle, deriv_const, energy);
    
   
    // int numParticles = particles.size();
    // for (int i = 0; i < numParticles; i++) {
    //     for (int qidx = 0; qidx < 4; qidx++ ) {
    //         Vec3 deriv = qrot.quaternionRotate(qrot.quaternionInvert(fit_qrot.q), qrot.dQ0_2[i][qidx]);
    //         forces[particles[i]] += -deriv * deriv_const[qidx]/numParticles;   
            
    //     } 
    // }
      
    
        
    return  acos(-unit_vec[1])*180/pi;
}


double ReferenceCalcPolaranglesForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    return calculateIxn(posData, forceData);
}

void ReferenceCalcPolaranglesForceKernel::copyParametersToContext(ContextImpl& context, const PolaranglesForce& force) {
    if (referencePos.size() != force.getReferencePositions().size())
        throw OpenMMException("updateParametersInContext: The number of reference positions has changed");
    particles = force.getParticles();
    if (particles.size() == 0)
        for (int i = 0; i < referencePos.size(); i++)
            particles.push_back(i);
    referencePos = force.getReferencePositions();
    Vec3 center;
    for (int i : particles)
        center += referencePos[i];
    center /= particles.size();
    for (Vec3& p : referencePos)
        p -= center;
}