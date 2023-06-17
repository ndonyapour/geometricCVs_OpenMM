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

#include "CommonQuaternionKernels.h"
#include "CommonQuaternionKernelSources.h"
#include "openmm/common/BondedUtilities.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include "jama_eig.h"
#include <set>

using namespace QuaternionPlugin;
using namespace OpenMM;
using namespace std;


void CommonCalcQuaternionForceKernel::initialize(const System& system, const QuaternionForce& force) {
    qidx = force.getQidx();
    // Create data structures.

    ContextSelector selector(cc);
    bool useDouble = cc.getUseDoublePrecision();
    int elementSize = (useDouble ? sizeof(double) : sizeof(float));
    int numParticles = force.getParticles().size();
    if (numParticles == 0)
        numParticles = system.getNumParticles();
    referencePos.initialize(cc, system.getNumParticles(), 4*elementSize, "referencePos");
    particles.initialize<int>(cc, numParticles, "particles");
    buffer.initialize(cc, 9, elementSize, "buffer");
    eigval.initialize(cc, 4, elementSize, "eigval");
    eigvec.initialize(cc, 4, 4*elementSize, "eigvec");
    recordParameters(force);
    info = new CommonQuaternionForceInfo(force);
    cc.addForce(info);
    // Create the kernels.
    // importnat variable
    blockSize = min(256, cc.getMaxThreadBlockSize());
    map<string, string> defines;
    defines["THREAD_BLOCK_SIZE"] = cc.intToString(blockSize);
    ComputeProgram program = cc.compileProgram(CommonQuaternionKernelSources::quaternionForce, defines);
    kernel1 = program->createKernel("computeQuaternionPart1");
    kernel2 = program->createKernel("computeQuaternionForces");
    kernel1->addArg();
    kernel1->addArg(cc.getPosq());
    kernel1->addArg(referencePos);
    kernel1->addArg(particles);
    kernel1->addArg(buffer);
    kernel2->addArg();
    kernel2->addArg();
    kernel2->addArg(cc.getPaddedNumAtoms());
    kernel2->addArg(cc.getPosq());
    kernel2->addArg(referencePos);
    kernel2->addArg(particles);
    kernel2->addArg(eigval);
    kernel2->addArg(eigvec);
    kernel2->addArg(cc.getLongForceBuffer());
}

void CommonCalcQuaternionForceKernel::recordParameters(const QuaternionForce& force) {
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

    // Record the sum of the norms of the reference positions.

    sumNormRef = 0.0;
    for (int i : particleVec) {
        Vec3 p = centeredPositions[i];
        sumNormRef += p.dot(p);
    }
}

double CommonCalcQuaternionForceKernel::execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) {
    ContextSelector selector(cc);
    if (cc.getUseDoublePrecision())
        return executeImpl<double>(context);
    return executeImpl<float>(context);
}

template <class REAL>
double CommonCalcQuaternionForceKernel::executeImpl(OpenMM::ContextImpl& context) {
    // Execute the first kernel.

    int numParticles = particles.getSize();
    kernel1->setArg(0, numParticles);
    kernel1->execute(blockSize, blockSize);

    // Download the Correlation matrix, build the S matrix, and find the maximum eigenvalue
    // and eigenvector.

    vector<REAL> C;
    buffer.download(C);

    // JAMA::Eigenvalue may run into an infinite loop if we have any NaN
    for (int i = 0; i < 9; i++) {
        if (C[i] != C[i])
            throw OpenMMException("NaN encountered during Quaternion force calculation");
    }

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

    JAMA::Eigenvalue<double> eigen(S);
    Array2D<double> S_eigvec;
    Array1D<double> S_eigval;
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
    for (int i=0;i<4;i++) {
        dot=0.0;
        for (int j=0;j<4;j++) {
            dot += normquat[j] * S_eigvec[i][j];
        }
        if (dot < 0.0)
            for (int j=0;j<4;j++)
                S_eigvec[i][j] = -S_eigvec[i][j];
    }

    // inverse eigen vectrs
    // inverse for quaternion inv(q) = (q0, -q1, -q, -q3)
    for (int i=0;i<4;i++) {
        for (int j=1;j<4;j++)
            S_eigvec[i][j] = -S_eigvec[i][j];
    }

    vector<REAL> eigval_buffer = {static_cast<REAL>(S_eigval[0]), static_cast<REAL>(S_eigval[1]),
                                  static_cast<REAL>(S_eigval[2]), static_cast<REAL>(S_eigval[3])};
    vector<mm_double4> eigvec_buffer;

    for (int i=0;i<4;i++)
        eigvec_buffer.push_back(mm_double4(S_eigvec[i][0], S_eigvec[i][1], S_eigvec[i][2], S_eigvec[i][3]));

    // if true, automatic conversions between single and double
    //                  precision will be performed as necessary
    eigval.upload(eigval_buffer);
    eigvec.upload(eigvec_buffer, true);
    kernel2->setArg(0, numParticles);
    kernel2->setArg(1, qidx);
    kernel2->execute(numParticles);
    return S_eigvec[0][qidx];
}

void CommonCalcQuaternionForceKernel::copyParametersToContext(OpenMM::ContextImpl& context, const QuaternionForce& force) {
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
