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

#include "EuleranglesForce.h"
#include "internal/EuleranglesForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace EuleranglesPlugin;
using namespace OpenMM;
using namespace std;

EuleranglesForce::EuleranglesForce(const vector<Vec3>& referencePositions, const vector<int>& particles, 
                                   const vector<int>& fitting_particles, const string& angle):
        referencePositions(referencePositions), particles(particles), fitting_particles(fitting_particles), angle(angle){
}

void EuleranglesForce::setReferencePositions(const std::vector<Vec3>& positions) {
    referencePositions = positions;
}

void EuleranglesForce::setParticles(const std::vector<int>& particles) {
    this->particles = particles;
}

void EuleranglesForce::setFittingParticles(const std::vector<int>& fitting_particles) {
    this->fitting_particles = fitting_particles;
}
void EuleranglesForce::setAngle(std::string& angle) {
    this->angle = angle;
}
void EuleranglesForce::updateParametersInContext(Context& context) {
    dynamic_cast<EuleranglesForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

ForceImpl* EuleranglesForce::createImpl() const {
    return new EuleranglesForceImpl(*this);
}
