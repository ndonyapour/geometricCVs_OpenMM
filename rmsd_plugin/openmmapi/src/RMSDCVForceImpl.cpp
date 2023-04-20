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

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "internal/RMSDCVForceImpl.h"
#include "RMSDCVKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <map>
#include <set>
#include <sstream>

using namespace RMSDCVPlugin;
using namespace OpenMM;
using namespace std;

RMSDCVForceImpl::RMSDCVForceImpl(const RMSDCVForce& owner) : owner(owner) {
}

RMSDCVForceImpl::~RMSDCVForceImpl() {
}

void RMSDCVForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcRMSDCVForceKernel::Name(), context);

    // Check for errors in the specification of particles.
    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    if (owner.getReferencePositions().size() != numParticles)
        throw OpenMMException("RMSDCVForce: Number of reference positions does not equal number of particles in the System");
    set<int> particles;
    for (int i : owner.getParticles()) {
        if (i < 0 || i >= numParticles) {
            stringstream msg;
            msg << "RMSDCVForce: Illegal particle index for RMSDCV: ";
            msg << i;
            throw OpenMMException(msg.str());
        }
        if (particles.find(i) != particles.end()) {
            stringstream msg;
            msg << "RMSDCVForce: Duplicated particle index for RMSDCV: ";
            msg << i;
            throw OpenMMException(msg.str());
        }
        particles.insert(i);
    }
    kernel.getAs<CalcRMSDCVForceKernel>().initialize(context.getSystem(), owner);
}

double RMSDCVForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcRMSDCVForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> RMSDCVForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcRMSDCVForceKernel::Name());
    return names;
}

void RMSDCVForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcRMSDCVForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

