#ifndef OPENMM_RMSDCVFORCE_H_
#define OPENMM_RMSDCVFORCE_H_

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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <vector>
#include "internal/windowsExportRMSDCV.h"


using namespace OpenMM;
namespace RMSDCVPlugin {

/**
 * This is a force whose energy equals the root mean squared deviation (RMSDCV)
 * between the current coordinates and a reference structure.  It is intended for
 * use with CustomCVForce.  You will not normally want a force that exactly equals
 * the RMSDCV, but there are many situations where it is useful to have a restraining
 * or biasing force that depends on the RMSDCV in some way.
 * 
 * The force is computed by first aligning the particle positions to the reference
 * structure, then computing the RMSDCV between the aligned positions and the reference.
 * The computation can optionally be done based on only a subset of the particles
 * in the system.
 */

class OPENMM_EXPORT RMSDCVForce : public OpenMM::Force {
public:
    /**
     * Create an RMSDCVForce.
     *
     * @param referencePositions  the reference positions to compute the deviation
     *                            from.  The length of this vector must equal the
     *                            number of particles in the system, even if not
     *                            all particles are used in computing the RMSDCV.
     * @param particles           the indices of the particles to use when computing
     *                            the RMSDCV.  If this is empty (the default), all
     *                            particles in the system will be used.
     */
    explicit RMSDCVForce(const std::vector<Vec3>& referencePositions,
                       const std::vector<int>& particles=std::vector<int>());
    /**
     * Get the reference positions to compute the deviation from.
     */
    const std::vector<Vec3>& getReferencePositions() const {
        return referencePositions;
    }
    /**
     * Set the reference positions to compute the deviation from.
     */
    void setReferencePositions(const std::vector<Vec3>& positions);
    /**
     * Get the indices of the particles to use when computing the RMSDCV.  If this
     * is empty, all particles in the system will be used.
     */
    const std::vector<int>& getParticles() const {
        return particles;
    }
    /**
     * Set the indices of the particles to use when computing the RMSDCV.  If this
     * is empty, all particles in the system will be used.
     */
    void setParticles(const std::vector<int>& particles);
    /**
     * Update the reference positions and particle indices in a Context to match those stored
     * in this Force object.  This method provides an efficient method to update certain parameters
     * in an existing Context without needing to reinitialize it.  Simply call setReferencePositions()
     * and setParticles() to modify this object's parameters, then call updateParametersInContext()
     * to copy them over to the Context.
     */
    void updateParametersInContext(OpenMM::Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const {
        return false;
    }
protected:
   OpenMM::ForceImpl* createImpl() const;
private:
    std::vector<Vec3> referencePositions;
    std::vector<int> particles;
};

} // namespace RMSDCVPlugin

#endif /*OPENMM_RMSDCVFORCE_H_*/
