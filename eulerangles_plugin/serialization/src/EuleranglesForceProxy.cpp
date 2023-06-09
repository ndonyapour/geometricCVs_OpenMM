/* -------------------------------------------------------------------------- *
 *                                OpenMMEulerangles                                 *
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

#include "EuleranglesForceProxy.h"
#include "EuleranglesForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <sstream>
#include <string>

using namespace EuleranglesPlugin;
using namespace OpenMM;
using namespace std;

EuleranglesForceProxy::EuleranglesForceProxy() : SerializationProxy("EuleranglesForce") {
}
void EuleranglesForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 0);
    const EuleranglesForce& force = *reinterpret_cast<const EuleranglesForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("angle", force.get_Angle());
    SerializationNode& positionsNode = node.createChildNode("ReferencePositions");
    for (const Vec3& pos : force.getReferencePositions())
       positionsNode.createChildNode("Position").setDoubleProperty("x", pos[0]).setDoubleProperty("y", pos[1]).setDoubleProperty("z", pos[2]);
    SerializationNode& particlesNode = node.createChildNode("Particles");
    for (int i : force.getParticles())
       particlesNode.createChildNode("Particle").setIntProperty("index", i);
    
    SerializationNode& fittingparticlesNode = node.createChildNode("FittingParticles");
    for (int i : force.getFittingParticles())
       fittingparticlesNode.createChildNode("Particle").setIntProperty("index", i);      
    
}

void* EuleranglesForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version != 0)
        throw OpenMMException("Unsupported version number");
    EuleranglesForce* force = NULL;
    try {
        vector<Vec3> positions;
        for (auto& pos : node.getChildNode("ReferencePositions").getChildren())
            positions.push_back(Vec3(pos.getDoubleProperty("x"), pos.getDoubleProperty("y"), pos.getDoubleProperty("z")));
        vector<int> particles;
        for (auto& particle : node.getChildNode("Particles").getChildren())
            particles.push_back(particle.getIntProperty("index"));
        vector<int> fitting_particles;
        for (auto& particle : node.getChildNode("FittingParticles").getChildren())
            fitting_particles.push_back(particle.getIntProperty("index"));
        string angle;
        angle = node.getStringProperty("angle");
        force = new EuleranglesForce(positions, particles, fitting_particles, angle);
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        return force;
    }
    catch (...) {
        if (force != NULL)
            delete force;
        throw;
    }
}