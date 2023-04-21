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

#include "ReferenceQuaternionKernels.h"
#include "QuaternionForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/Vec3.h"

#include "jama_eig.h"
#include "Quaternion.h"

using namespace QuaternionPlugin;
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

ReferenceCalcQuaternionForceKernel::~ReferenceCalcQuaternionForceKernel() {
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

void ReferenceCalcQuaternionForceKernel::initialize(const System& system, const QuaternionForce& force) {
    particles = force.getParticles();
    if (particles.size() == 0)
        for (int i = 0; i < system.getNumParticles(); i++)
            particles.push_back(i);
    referencePos = force.getReferencePositions();
    Vec3 center;
    for (int i : particles)
        center += referencePos[i];
    center /= particles.size();
    for (Vec3& p : referencePos)
        p -= center; 
   
}

double ReferenceCalcQuaternionForceKernel::calculateIxn(vector<OpenMM::Vec3>& atomCoordinates, vector<OpenMM::Vec3>& forces) const {
    // Compute the Quaternion and its gradient using the algorithm described in Coutsias et al,
    // "Using quaternions to calculate Quaternion" (doi: 10.1002/jcc.20110).  First subtract
    // the centroid from the atom positions.  The reference positions have already been centered.
    std::vector<Vec3> refpos, pos, centered_refpos, centered_pos;
    //currpos1.resize(refpos1.size(), Vector);
    for (int i : particles ){
        refpos.push_back(referencePos[i]);
        pos.push_back(atomCoordinates[i]);
    }
    
    // centering 
    centered_refpos = shiftbyCOG(refpos);
    centered_pos = shiftbyCOG(pos);
    
    // calculating q 
    Quaternion qrot;
    std::vector<double> normquat = {1.0, 0.0, 0.0, 0.0}; 
    qrot.request_group2_gradients(pos.size());
    qrot.calc_optimal_rotation(centered_refpos, centered_pos, normquat);
    
    // claculate forces 
    int qidx = 0;
    int numParticles = particles.size();
    for (int i = 0; i < numParticles; i++) 
        forces[particles[i]] += qrot.dQ0_2[i][qidx] * 1/numParticles; 
        
    return qrot.q0;
}


double ReferenceCalcQuaternionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    return calculateIxn(posData, forceData);
}

void ReferenceCalcQuaternionForceKernel::copyParametersToContext(ContextImpl& context, const QuaternionForce& force) {
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