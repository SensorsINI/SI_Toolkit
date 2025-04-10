#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Include the weights and biases stored as arrays in separate .c file
#include "network.h"
#include "network_parameters.c"

#if IS_GRU

static float h1[GRU1_UNITS];
static float h2[GRU2_UNITS];

void InitializeGRUStates(void) {
    // Copy initial_h1 into h1
    for (int i = 0; i < GRU1_UNITS; i++) {
        h1[i] = initial_h1[i];
    }
    // Copy initial_h2 into h2
    for (int i = 0; i < GRU2_UNITS; i++) {
        h2[i] = initial_h2[i];
    }
}
#endif

static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static inline void applyTanh(float* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = tanhf(x[i]);  // Apply tanh activation function
    }
}

static void matMul(const float* matrix, const float* vec, float* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0f;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vec[j];  // Matrix multiplication (matrix * vector)
        }
    }
}

static void addBias(const float* bias, float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] += bias[i];  // Adding bias to each element in the vector
    }
}

//--------------------------------------
// GRU forward pass for one GRU layer
// Keras ordering: [z, r, h] along last dimension
//--------------------------------------
static void GRU_Forward(
    const float* inputs,     // shape: [input_dim]
    float* h,                // shape: [units] (hidden state)
    int input_dim,
    int units,
    const float* kernel,     // shape: [input_dim, 3*units]
    const float* recurrent,  // shape: [units, 3*units]
    const float* bias        // shape: [6*units] (split input+recurrent bias)
)
{
    float h_prev[units];
    memcpy(h_prev, h, sizeof(float) * units);

    const int base_input = 0;
    const int base_recur = 3 * units;

    for (int i = 0; i < units; i++)
    {
        //---------------------------
        //   Update gate (z)
        //---------------------------
        float x_z = 0.0f, h_z = 0.0f;
        for (int d = 0; d < input_dim; d++) {
            x_z += inputs[d] * kernel[d*(3*units) + 0*units + i];
        }
        x_z += bias[base_input + 0*units + i];

        for (int u = 0; u < units; u++) {
            h_z += h_prev[u] * recurrent[u*(3*units) + 0*units + i];
        }
        h_z += bias[base_recur + 0*units + i];

        float z_gate = sigmoidf(x_z + h_z);

        //---------------------------
        //   Reset gate (r)
        //---------------------------
        float x_r = 0.0f, h_r = 0.0f;
        for (int d = 0; d < input_dim; d++) {
            x_r += inputs[d] * kernel[d*(3*units) + 1*units + i];
        }
        x_r += bias[base_input + 1*units + i];

        for (int u = 0; u < units; u++) {
            h_r += h_prev[u] * recurrent[u*(3*units) + 1*units + i];
        }
        h_r += bias[base_recur + 1*units + i];

        float r_gate = sigmoidf(x_r + h_r);

        //---------------------------
        //   Candidate activation (h~)
        //---------------------------
        float x_h = 0.0f, h_h = 0.0f;
        for (int d = 0; d < input_dim; d++) {
            x_h += inputs[d] * kernel[d*(3*units) + 2*units + i];
        }
        x_h += bias[base_input + 2*units + i];

        for (int u = 0; u < units; u++) {
            h_h += (h_prev[u] * r_gate) * recurrent[u*(3*units) + 2*units + i];
        }
        // Multiply the recurrent bias by r_gate *****
        h_h += r_gate * bias[base_recur + 2*units + i];

        float hh = tanhf(x_h + h_h);

        //---------------------------
        //   New hidden state
        //---------------------------
        float old_h = h_prev[i];
        h[i] = z_gate * old_h + (1.0f - z_gate) * hh;
    }
}

//--------------------------------------
// Single forward pass
//--------------------------------------
void C_Network_Evaluate(float* inputs, float* outputs) {
#if IS_GRU
    //--------------------------------------
    // GRU-based forward pass (two GRU layers + final Dense)
    //--------------------------------------

    GRU_Forward(inputs, h1, INPUT_SIZE, GRU1_UNITS,
                gru1_kernel, gru1_recurrent_kernel, gru1_bias);

    GRU_Forward(h1, h2, GRU1_UNITS, GRU2_UNITS,
                gru2_kernel, gru2_recurrent_kernel, gru2_bias);

    matMul(weights3, h2, outputs, LAYER3_SIZE, GRU2_UNITS);
    addBias(bias3, outputs, LAYER3_SIZE);

#else
    //--------------------------------------
    // Original feed-forward network code
    //--------------------------------------

    float* layer1 = (float*) malloc(LAYER1_SIZE * sizeof(float));
    matMul(weights1, inputs, layer1, LAYER1_SIZE, INPUT_SIZE);
    addBias(bias1, layer1, LAYER1_SIZE);
    applyTanh(layer1, LAYER1_SIZE);

    float* layer2 = (float*) malloc(LAYER2_SIZE * sizeof(float));
    matMul(weights2, layer1, layer2, LAYER2_SIZE, LAYER1_SIZE);
    addBias(bias2, layer2, LAYER2_SIZE);
    applyTanh(layer2, LAYER2_SIZE);

    matMul(weights3, layer2, outputs, LAYER3_SIZE, LAYER2_SIZE);
    addBias(bias3, outputs, LAYER3_SIZE);

    free(layer1);
    free(layer2);
#endif
}
