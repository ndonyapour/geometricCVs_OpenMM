#ifndef REFERENCE_Quaternion_KERNELS_H_
#define REFERENCE_Quaternion_KERNELS_H_

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

#include "QuaternionKernels.h"
#include "openmm/Platform.h"
#include <vector>

namespace QuaternionPlugin {

class ReferenceCalcQuaternionForceKernel : public CalcQuaternionForceKernel {
private:
    std::vector<OpenMM::Vec3> referencePos;
    std::vector<int> particles;

public:
    /**
     * Constructor
     */
    ReferenceCalcQuaternionForceKernel(std::string name, const OpenMM::Platform& platform) : CalcQuaternionForceKernel(name, platform) {
    }
      /**
     * Destructor
     */
    ~ReferenceCalcQuaternionForceKernel();

 /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the QuaternionForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const QuaternionForce& force);
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
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the QuaternionForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const QuaternionForce& force);
       /**
     * Calculate the interaction.
     * 
     * @param atomCoordinates    atom coordinates
     * @param forces             the forces are added to this
     * @return the energy of the interaction
     */
   double calculateIxn(std::vector<OpenMM::Vec3>& atomCoordinates, std::vector<OpenMM::Vec3>& forces) const;
};

} // namespace QuaternionPlugin

#endif // __ReferenceCalcQuaternionForceKernel_H__

