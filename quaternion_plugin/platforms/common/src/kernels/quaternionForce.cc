// This file contains kernels to compute the Quaternion and its gradient using the algorithm described
// in Coutsias et al, "Using quaternions to calculate Quaternion" (doi: 10.1002/jcc.20110).

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
 * Perform the first step of computing the Quaternion.  This is executed as a single work group.
 */
KERNEL void computeQuaternionPart1(int numParticles, GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT referencePos,
        GLOBAL const int* RESTRICT particles, GLOBAL real* buffer) {
    LOCAL volatile real temp[THREAD_BLOCK_SIZE];

    // Compute the center of the particle positions.
    
    real3 center = make_real3(0);
    for (int i = LOCAL_ID; i < numParticles; i += LOCAL_SIZE)
        center += trimTo3(posq[particles[i]]);
    center.x = reduceValue(center.x, temp)/numParticles;
    center.y = reduceValue(center.y, temp)/numParticles;
    center.z = reduceValue(center.z, temp)/numParticles;
    
    // Compute the correlation matrix.
    
    real R[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    real sum = 0;
    for (int i = LOCAL_ID; i < numParticles; i += LOCAL_SIZE) {
        int index = particles[i];
        real3 pos = trimTo3(posq[index]) - center;
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
        buffer[9] = sum;
        buffer[10] = center.x;
        buffer[11] = center.y;
        buffer[12] = center.z;
    }
}

/**
 * Apply forces based on the Quaternion.
 */
KERNEL void computeQuaternionForces(int numParticles, int qidx, int paddedNumAtoms, GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT referencePos,
        GLOBAL const int* RESTRICT particles, GLOBAL real* poscenter, GLOBAL real* eigval, const real4* RESTRICT eigvec, GLOBAL mm_long* RESTRICT forceBuffers) {
    real3 center = make_real3(poscenter[0], poscenter[1], poscenter[2]);
    real scale = 1 / (real) (numParticles);
    real3 ds_2[4][4];
    real L0 = eigval[0], L1 = eigval[1], L2 = eigval[2], L3 = eigval[3];
    real Q0[4] = {eigvec[0].x,  eigvec[0].y, eigvec[0].z, eigvec[0].w};
    real Q1[4] = {eigvec[1].x,  eigvec[1].y, eigvec[1].z, eigvec[1].w};
    real Q2[4] = {eigvec[2].x,  eigvec[2].y, eigvec[2].z, eigvec[2].w};
    real Q3[4] = {eigvec[3].x,  eigvec[3].y, eigvec[3].z, eigvec[3].w};
    
    for (int i = GLOBAL_ID; i < numParticles; i += GLOBAL_SIZE) {
        int index = particles[i];
        //real3 pos = trimTo3(posq[index]) - center;
        real3 refPos = trimTo3(referencePos[index]);
        real rx = refPos.x, ry = refPos.y, rz = refPos.z;
        ds_2[0][0] = make_real3(  rx,  ry,  rz);
        ds_2[1][0] = make_real3( 0.0, -rz,  ry);
        ds_2[0][1] = ds_2[1][0];
        ds_2[2][0] = make_real3(  rz, 0.0, -rx);
        ds_2[0][2] = ds_2[2][0];
        ds_2[3][0] = make_real3( -ry,  rx, 0.0);
        ds_2[0][3] = ds_2[3][0];
        ds_2[1][1] = make_real3(  rx, -ry, -rz);
        ds_2[2][1] = make_real3(  ry,  rx, 0.0);
        ds_2[1][2] = ds_2[2][1];
        ds_2[3][1] = make_real3(  rz, 0.0,  rx);
        ds_2[1][3] = ds_2[3][1];
        ds_2[2][2] = make_real3( -rx,  ry, -rz);
        ds_2[3][2] = make_real3( 0.0,  rz,  ry);
        ds_2[2][3] = ds_2[3][2];
        ds_2[3][3] = make_real3( -rx, -ry,  rz);
        
        real3 dl0_2;
        real3 dq0_2[4];

        for (unsigned i = 0; i < 4; i++) {
            for (unsigned j = 0; j < 4; j++) {
                dl0_2 += -1 * (Q0[i] * ds_2[i][j] * Q0[j]);
            }
        }
        for (int p=0; p<4; p++) {
            for (int i=0 ;i<4; i++) {
                for (int j=0; j<4; j++) {
                    dq0_2[p] += -1 * ((Q1[i] * ds_2[i][j] * Q0[j]) / (L0-L1) * Q1[p] 
                                    + (Q2[i] * ds_2[i][j] * Q0[j]) / (L0-L2) * Q2[p] 
                                    + (Q3[i] * ds_2[i][j] * Q0[j]) / (L0-L3) * Q3[p]);
                }
            }
        }
                                      
                                      
        real3 force = dq0_2[qidx] * scale;
        forceBuffers[index] += (mm_long) (force.x*0x100000000);
        forceBuffers[index+paddedNumAtoms] += (mm_long) (force.y*0x100000000);
        forceBuffers[index+2*paddedNumAtoms] += (mm_long) (force.z*0x100000000); //realToFixedPoint(force.z);
    }
}