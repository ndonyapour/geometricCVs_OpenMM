#ifndef OPENMM_PolaranglesFORCE_H_
#define OPENMM_PolaranglesFORCE_H_

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
#include <string>
#include "internal/windowsExportPolarangles.h"


using namespace OpenMM;
namespace PolaranglesPlugin {

/**
 * This is a force whose energy equals the root mean squared deviation (Polarangles)
 * between the current coordinates and a reference structure.  It is intended for
 * use with CustomCVForce.  You will not normally want a force that exactly equals
 * the Polarangles, but there are many situations where it is useful to have a restraining
 * or biasing force that depends on the Polarangles in some way.
 * 
 * The force is computed by first aligning the particle positions to the reference
 * structure, then computing the Polarangles between the aligned positions and the reference.
 * The computation can optionally be done based on only a subset of the particles
 * in the system.
 */

class OPENMM_EXPORT PolaranglesForce : public OpenMM::Force {
public:
    /**
     * Create an PolaranglesForce.
     *
     * @param referencePositions  the reference positions to compute the deviation
     *                            from.  The length of this vector must equal the
     *                            number of particles in the system, even if not
     *                            all particles are used in computing the Polarangles.
     * @param particles           the indices of the particles to use when computing
     *                            the Polarangles.  If this is empty (the default), all
     *                            particles in the system will be used.
     * @param fitting_particles   the indices of the fitting particles to use when computing
     *                            the Polarangles.  If this is empty (the default), no fitting 
     *                            will be applied.
     * @param angle               the angle type which can be one of these values "Theta", "Phi" or "Psi"
     */
    explicit PolaranglesForce(const std::vector<Vec3>& referencePositions,
                       const std::vector<int>& particles=std::vector<int>(), 
                       const std::vector<int>& fitting_particles=std::vector<int>(),
                       const std::string& angle="Theta");
    /**
     * Get the reference positions to compute the deviation from.
     */
    const std::vector<Vec3>& getReferencePositions() const {
        return referencePositions;
    }
     /**
     * Get the type of the euler angles to use when computing the Polarangles.  If this
     * is empty, the Theta angle will be used.
     */
    const std::string& get_Angle() const {
        return angle;
    }
    /**
     * Set the reference positions to compute the deviation from.
     */
    void setReferencePositions(const std::vector<Vec3>& positions);
    /**
     * Get the indices of the particles to use when computing the Polarangles.  If this
     * is empty, all particles in the system will be used.
     */
    const std::vector<int>& getParticles() const {
        return particles;
    }
    /**
     * Get the indices of the fitting particles to use when computing the Polarangles.  If this
     * is empty, no fitting will be applied.
     */
    const std::vector<int>& getFittingParticles() const {
        return fitting_particles;
    }
    /**
     * Set the indices of the particles to use when computing the Polarangles.  If this
     * is empty, all particles in the system will be used.
     */
    void setParticles(const std::vector<int>& particles);
     /**
     * Set the indices of the fitting particles to use when computing the Polarangles.  If this
     * is empty, no fitting will be applied.
     */
    void setFittingParticles(const std::vector<int>& fitting_particles);
    /**
     * Set the index of Q to use when computing the Polarangles.  If this
     * is empty, all particles in the system will be used.
     */
    void setAngle(std::string& angle);
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
    std::vector<int> fitting_particles;
    std::string angle;
};

} // namespace PolaranglesPlugin

#endif /*OPENMM_PolaranglesFORCE_H_*/
