%module Quaternionplugin
%include "factory.i"

%{
#include "QuaternionForce.h"
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

%pythonappend QuaternionPlugin::QuaternionForce::getReferencePositions() const %{
    val = unit.Quantity(val, unit.nanometer)
%}


%pythonappend RMSDPlugin::RMSDForce::RMSDForce(const std::vector<Vec3> &referencePositions, const std::vector<int> &particles=std::vector<int>())%{
    for arg in args:
        if 'numpy' in sys.modules and isinstance(arg, numpy.ndarray):
            arg = arg.tolist()
        
%}

// %typemap(in) std::vector<int> {
//     PyObject *obj = $input;
//     if (!PyList_Check(obj)) {
//         PyErr_SetString(PyExc_TypeError, "expected a list");
//         return NULL;
//     }
//     Py_ssize_t n = PyList_Size(obj);
//     std::vector<int> vec(n);
//     for (Py_ssize_t i = 0; i < n; i++) {
//         PyObject *item = PyList_GetItem(obj, i);
//         if (!PyLong_Check(item)) {
//             PyErr_SetString(PyExc_TypeError, "expected a list of ints");
//             return NULL;
//         }
//         vec[i] = PyLong_AsLong(item);
//     }
//     $1 = vec;
// }

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace QuaternionPlugin {

class QuaternionForce : public OpenMM::Force {
public:

    QuaternionForce(const std::vector<Vec3> &referencePositions, const std::vector<int> &particles=std::vector<int>());
    virtual bool usesPeriodicBoundaryConditions() const;
    void setParticles(const std::vector<int> &particles);    
    void setReferencePositions(const std::vector<Vec3> &positions);
    %apply OpenMM::Context & OUTPUT {OpenMM::Context & context };    
    void updateParametersInContext(OpenMM::Context& context);
    %clear Context & context;
    const std::vector<int>& getParticles() const;
    const std::vector<Vec3>& getReferencePositions() const;



   /*
     * Add methods for casting a Force to an QuaternionForce.
    */
    %extend {
        static QuaternionPlugin::QuaternionForce& cast(OpenMM::Force& force) {
            return dynamic_cast<QuaternionPlugin::QuaternionForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<QuaternionPlugin::QuaternionForce*>(&force) != NULL);
        }
    }

};

}
