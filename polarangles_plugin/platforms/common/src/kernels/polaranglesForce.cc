// This file contains kernels to compute the Polarangles and its gradient using the algorithm described
// in Coutsias et al, "Using polarangless to calculate Polarangles" (doi: 10.1002/jcc.20110).

// KERNEL __forceinline__ long realToFixedPoint(real x) {
//     return (long) (x * 0x100000000);
// }
// __device__ inline long long realToFixedPoint(real x) {
// 	    return static_cast<long long>(x * 0x100000000);
// }

/**
 * Sum a value over all threads.
 */
DEVICE real reduceValue(real value, LOCAL_ARG volatile real* temp) {
    const int thread = LOCAL_ID;
    SYNC_THREADS;
    temp[thread] = value;
    SYNC_THREADS;
    for (int step = 1; step < 32; step *= 2) {
        if (thread+step < LOCAL_SIZE && thread%(2*step) == 0)
            temp[thread] = temp[thread] + temp[thread+step];
        SYNC_WARPS;
    }
    for (int step = 32; step < LOCAL_SIZE; step *= 2) {
        if (thread+step < LOCAL_SIZE && thread%(2*step) == 0)
            temp[thread] = temp[thread] + temp[thread+step];
        SYNC_THREADS;
    }
    return temp[0];
}

/**
* calculate COG 
*/
KERNEL void computecCOG(int numParticles, GLOBAL const real4* RESTRICT posq, GLOBAL const int* RESTRICT particles, 
                        const GLOBAL real* poscenter, const GLOBAL real* qrot, GLOBAL real* centr_buffer) {
    
    LOCAL volatile real temp[THREAD_BLOCK_SIZE];
    // Compute the center of the particle positions.
    real3 center =  make_real3(poscenter[0], poscenter[1], poscenter[2]);
    real q0 = qrot[0];
    real3 vq = make_real3(qrot[1], qrot[2], qrot[3]);
    real3 rotcenter = make_real3(0, 0, 0); 
 
    for (int i = LOCAL_ID; i < numParticles; i += LOCAL_SIZE){
        real3 pos = trimTo3(posq[particles[i]]) - center;
        // rotate 
        real3 a, b;
        a = cross(pos, vq) + q0 * pos;        
        b = cross(a, vq);
        pos = b + b + pos;
        rotcenter += pos;
        
    }
    rotcenter.x = reduceValue(rotcenter.x, temp)/numParticles;
    rotcenter.y = reduceValue(rotcenter.y, temp)/numParticles;
    rotcenter.z = reduceValue(rotcenter.z, temp)/numParticles;
    if (LOCAL_ID == 0) {
        centr_buffer[0] = rotcenter.x;
        centr_buffer[1] = rotcenter.y;
        centr_buffer[2] = rotcenter.z;
    }
}


/**
 * Perform the first step of computing the Polarangles.  This is executed as a single work group.
*/
KERNEL void computePolaranglesPart1(int numParticles, bool usecenter, bool rotate, GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT referencePos,
        GLOBAL const int* RESTRICT particles, const GLOBAL real* poscenter, const GLOBAL real* qrot, GLOBAL real* buffer) {
    LOCAL volatile real temp[THREAD_BLOCK_SIZE];

    // Compute the center of the particle positions.
    real3 center = make_real3(0, 0, 0); 
    if (!usecenter) {
        for (int i = LOCAL_ID; i < numParticles; i += LOCAL_SIZE)
            center += trimTo3(posq[particles[i]]);
        center.x = reduceValue(center.x, temp)/numParticles;
        center.y = reduceValue(center.y, temp)/numParticles;
        center.z = reduceValue(center.z, temp)/numParticles;
    }
    else
        center = make_real3(poscenter[0], poscenter[1], poscenter[2]);
    
    //printf("Device %f \t %f \t %f\n", poscenter[0], poscenter[1], poscenter[2]);
    // Compute the correlation matrix.
    
    real R[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    real sum = 0;
    real q0 = qrot[0];
    real3 vq = make_real3(qrot[1], qrot[2], qrot[3]);
    for (int i = LOCAL_ID; i < numParticles; i += LOCAL_SIZE) {
        int index = particles[i];
        real3 pos = trimTo3(posq[index]) - center;
        if (rotate) {
            real3 a, b;
            a = cross(pos, vq) + q0 * pos;        
            b = cross(a, vq);
            pos = b + b + pos;
        }
            // do q rotate 
            
        real3 refPos = trimTo3(referencePos[index]);
        R[0][0] += pos.x*refPos.x;
        R[0][1] += pos.x*refPos.y;
        R[0][2] += pos.x*refPos.z;
        R[1][0] += pos.y*refPos.x;
        R[1][1] += pos.y*refPos.y;
        R[1][2] += pos.y*refPos.z;
        R[2][0] += pos.z*refPos.x;
        R[2][1] += pos.z*refPos.y;
        R[2][2] += pos.z*refPos.z;
        sum += dot(pos, pos);
    }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R[i][j] = reduceValue(R[i][j], temp);
    sum = reduceValue(sum, temp);

    // Copy everything into the output buffer to send back to the host.
    
    if (LOCAL_ID == 0) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                buffer[3*i+j] = R[i][j];
        buffer[9] = center.x;
        buffer[10] = center.y;
        buffer[11] = center.z;
    }
}



/**
 * Apply forces based on the Polarangles.
 */
KERNEL void computePolaranglesForces(int numParticles, int paddedNumAtoms, GLOBAL const int* RESTRICT particles, GLOBAL real4* derive_matrix, 
        GLOBAL real* anglederiv, GLOBAL const real* RESTRICT qrot_deriv, GLOBAL mm_long* RESTRICT forceBuffers) {
    //real3 center = make_real3(poscenter[0], poscenter[1], poscenter[2]);
    real scale = 1 / (real) (numParticles);
 
    //printf("***********************************\n");
    //printf("%d \t %d \t %d \t\n", GLOBAL_ID, GLOBAL_SIZE, numParticles);
    real q0 = qrot_deriv[0];
    real3 vq = make_real3(qrot_deriv[1], qrot_deriv[2], qrot_deriv[3]);
    for (int i = GLOBAL_ID; i < numParticles; i += GLOBAL_SIZE) {
        int index = particles[i];
        real3 dvec[4];
        duvec[0] = make_real3(0, 0, 0); duvec[1] = make_real3(0, 0, 0); duvec[2] = make_real3(0, 0, 0); 
        for (int p=0; p<3; p++){
            real3 mderiv = trimTo3(derive_matrix[p]);
            real3 a, b;
            a = cross(mderiv, vq) + q0 * mderiv;
            b = cross(a, vq);
            dvec[p] = b + b + mderiv; 
            }          
    
        real3 force = make_real3(0, 0, 0);
        for(int i=0; i<3; i++)
            force += -dvec[i] * anglederiv[i] * scale; 

        forceBuffers[index] += (mm_long) (force.x*0x100000000);
        forceBuffers[index+paddedNumAtoms] += (mm_long) (force.y*0x100000000);
        forceBuffers[index+2*paddedNumAtoms] += (mm_long) (force.z*0x100000000); //realToFixedPoint(force.z);
    }
}