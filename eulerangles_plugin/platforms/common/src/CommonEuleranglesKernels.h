#ifndef COMMON_Eulerangles_KERNELS_H_
#define COMMON_Eulerangles_KERNELS_H_

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

#include "EuleranglesKernels.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeArray.h"
#include <set>

using namespace std;

namespace EuleranglesPlugin {
class CommonEuleranglesForceInfo : public ComputeForceInfo {
public:
    CommonEuleranglesForceInfo(const EuleranglesForce& force) : force(force) {
        updateParticles();
    }
    void updateParticles() {
        particles.clear();
        for (int i : force.getParticles())
            particles.insert(i);
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        bool include1 = (particles.find(particle1) != particles.end());
        bool include2 = (particles.find(particle2) != particles.end());
        return (include1 == include2);
    }
private:
    const EuleranglesForce& force;
    set<int> particles;
};
};
namespace EuleranglesPlugin {

/**
 * This kernel is invoked by EuleranglesForce to calculate the forces acting on the system and the energy of the system.
 */
class CommonCalcEuleranglesForceKernel : public CalcEuleranglesForceKernel {
public:
    CommonCalcEuleranglesForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system) :
            CalcEuleranglesForceKernel(name, platform), hasInitializedKernel(false), cc(cc), system(system) {
    }


 /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the EuleranglesForce this kernel will be used for
     */
    void initialize(const System& system, const EuleranglesForce& force);
    /**
     * Record the reference positions and particle indices.
     */
    void recordParameters(const EuleranglesForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * This is the internal implementation of execute(), templatized on whether we're
     * using single or double precision.
     */
    template <class REAL>
    double executeImpl(OpenMM::ContextImpl& context);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the EuleranglesForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const EuleranglesForce& force);
private:
    bool hasInitializedKernel;
    OpenMM::ComputeContext& cc;
    const OpenMM::System& system;
    int blockSize, qidx;
    double sumNormRef;
    OpenMM::ComputeArray referencePos, particles, buffer, fit_referencePos, fit_particles, fit_buffer, fit_poscenter;
    OpenMM::ComputeArray eigval, eigvec, poscenter, qrot, qrot_deriv, anglederiv, forces;
    OpenMM::ComputeKernel kernel1, kernel2, kernel3, kernel4;
    CommonEuleranglesForceInfo* info;
    std::string angle;
    bool enable_fitting=false, enable_cpu=false;
    vector<Vec3> centeredPositions, fit_centeredPositions;
    vector<int> particleVec, fit_particleVec;


};

} // namespace EuleranglesPlugin

#endif /*COMMON_Eulerangles_KERNELS_H_*/
