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

using namespace QuaternionPlugin;
using namespace OpenMM;
using namespace std;


// ReferenceCalcQuaternionForceKernel::ReferenceCalcQuaternionForceKernel(vector<OpenMM::Vec3>& referencePos, vector<int>& particles) :
//         referencePos(referencePos), particles(particles) {
// }

ReferenceCalcQuaternionForceKernel::~ReferenceCalcQuaternionForceKernel() {
}

double ReferenceCalcQuaternionForceKernel::calculateIxn(vector<OpenMM::Vec3>& atomCoordinates, vector<OpenMM::Vec3>& forces) const {
    // Compute the Quaternion and its gradient using the algorithm described in Coutsias et al,
    // "Using quaternions to calculate Quaternion" (doi: 10.1002/jcc.20110).  First subtract
    // the centroid from the atom positions.  The reference positions have already been centered.
    
    int numParticles = particles.size();
    Vec3 center;
    for (int i : particles)
        center += atomCoordinates[i];
    center /= numParticles;
    vector<OpenMM::Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++)
        positions[i] = atomCoordinates[particles[i]]-center;
    
    // Compute the correlation matrix.
    
    double R[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < numParticles; k++) {
                int index = particles[k];
                R[i][j] += positions[k][i]*referencePos[index][j];
            }

    // Compute the F matrix.

    Array2D<double> F(4, 4);
    F[0][0] =  R[0][0] + R[1][1] + R[2][2];
    F[1][0] =  R[1][2] - R[2][1];
    F[2][0] =  R[2][0] - R[0][2];
    F[3][0] =  R[0][1] - R[1][0];

    F[0][1] =  R[1][2] - R[2][1];
    F[1][1] =  R[0][0] - R[1][1] - R[2][2];
    F[2][1] =  R[0][1] + R[1][0];
    F[3][1] =  R[0][2] + R[2][0];

    F[0][2] =  R[2][0] - R[0][2];
    F[1][2] =  R[0][1] + R[1][0];
    F[2][2] = -R[0][0] + R[1][1] - R[2][2];
    F[3][2] =  R[1][2] + R[2][1];

    F[0][3] =  R[0][1] - R[1][0];
    F[1][3] =  R[0][2] + R[2][0];
    F[2][3] =  R[1][2] + R[2][1];
    F[3][3] = -R[0][0] - R[1][1] + R[2][2];

    // Find the maximum eigenvalue and eigenvector.

    JAMA::Eigenvalue<double> eigen(F);
    Array1D<double> values;
    eigen.getRealEigenvalues(values);
    Array2D<double> vectors;
    eigen.getV(vectors);

    // Compute the Quaternion.
    
    double sum = 0.0;
    for (int i = 0; i < numParticles; i++) {
        int index = particles[i];
        sum += positions[i].dot(positions[i]) + referencePos[index].dot(referencePos[index]);
    }
    double msd = (sum-2*values[3])/numParticles;
    if (msd < 1e-20) {
        // The particles are perfectly aligned, so all the forces should be zero.
        // Numerical error can lead to NaNs, so just return 0 now.
        return 0.0;
    }
    double Quaternion = sqrt(msd);

    // Compute the rotation matrix.

    double q[] = {vectors[0][3], vectors[1][3], vectors[2][3], vectors[3][3]};
    double q00 = q[0]*q[0], q01 = q[0]*q[1], q02 = q[0]*q[2], q03 = q[0]*q[3];
    double q11 = q[1]*q[1], q12 = q[1]*q[2], q13 = q[1]*q[3];
    double q22 = q[2]*q[2], q23 = q[2]*q[3];
    double q33 = q[3]*q[3];
    double U[3][3] = {{q00+q11-q22-q33, 2*(q12-q03), 2*(q13+q02)},
                      {2*(q12+q03), q00-q11+q22-q33, 2*(q23-q01)},
                      {2*(q13-q02), 2*(q23+q01), q00-q11-q22+q33}};

    // Rotate the reference positions and compute the forces.
    
    for (int i = 0; i < numParticles; i++) {
        const Vec3& p = referencePos[particles[i]];
        Vec3 rotatedRef(U[0][0]*p[0] + U[1][0]*p[1] + U[2][0]*p[2],
                        U[0][1]*p[0] + U[1][1]*p[1] + U[2][1]*p[2],
                        U[0][2]*p[0] + U[1][2]*p[1] + U[2][2]*p[2]);
        forces[particles[i]] -= (positions[i]-rotatedRef) / (Quaternion*numParticles);
    }
    return Quaternion;
}

void ReferenceCalcQuaternionForceKernel::initialize(const System& system, const QuaternionForce& force) {
    // Initialize bond parameters.
    
    // int numBonds = force.getNumBonds();
    // particle1.resize(numBonds);
    // particle2.resize(numBonds);
    // length.resize(numBonds);
    // k.resize(numBonds);
   
}

double ReferenceCalcQuaternionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // vector<RealVec>& pos = extractPositions(context);
    // vector<RealVec>& force = extractForces(context);
    // int numBonds = particle1.size();
    double energy = 0;

    return energy;
}

void ReferenceCalcQuaternionForceKernel::copyParametersToContext(ContextImpl& context, const QuaternionForce& force) {
    // if (force.getNumBonds() != particle1.size())
    //     throw OpenMMException("updateParametersInContext: The number of Quaternion bonds has changed");
    // for (int i = 0; i < force.getNumBonds(); i++) {
    //     int p1, p2;
    //     force.getBondParameters(i, p1, p2, length[i], k[i]);
    //     if (p1 != particle1[i] || p2 != particle2[i])
    //         throw OpenMMException("updateParametersInContext: A particle index has changed");
    // }
}