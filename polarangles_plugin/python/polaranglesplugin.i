%module Polaranglesplugin
%include "factory.i"

%{
#include "PolaranglesForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include "openmm/Vec3.h"
%}

%{
#include <numpy/arrayobject.h>
%}

%include "std_string.i"
%include "std_vector.i"
%include "typemaps.i"
%include "header.i"

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"



//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */
using namespace OpenMM;
//%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%pythoncode %{
import openmm as mm
import simtk.unit as unit
%}

/*
 * Add units to function outputs.
*/

%pythonappend PolaranglesPlugin::PolaranglesForce::getReferencePositions() const %{
    val = unit.Quantity(val, unit.nanometer)
%}

%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace PolaranglesPlugin {

class PolaranglesForce : public OpenMM::Force {
public:

    PolaranglesForce(const std::vector<Vec3> &referencePositions, const std::vector<int> &particles=std::vector<int>(), 
                     const std::vector<int> &fitting_particles=std::vector<int>(), const std::string &angle="Theta");
    virtual bool usesPeriodicBoundaryConditions() const;
    void setParticles(const std::vector<int> &particles);
    void setFittingParticles(const std::vector<int> &fitting_particles);    
    void setReferencePositions(const std::vector<Vec3> &positions);
    void setAngle(std::string& angle);
    %apply OpenMM::Context & OUTPUT {OpenMM::Context &context };    
    void updateParametersInContext(OpenMM::Context& context);
    %clear Context & context;
    const std::vector<int>& getParticles() const;
    const std::string& get_Angle() const;
    const std::vector<int>& getFittingParticles() const;
    const std::vector<Vec3>& getReferencePositions() const;



   /*
     * Add methods for casting a Force to an PolaranglesForce.
    */
    %extend {
        static PolaranglesPlugin::PolaranglesForce& cast(OpenMM::Force& force) {
            return dynamic_cast<PolaranglesPlugin::PolaranglesForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<PolaranglesPlugin::PolaranglesForce*>(&force) != NULL);
        }
    }

};

}
