// network.h
#ifndef NETWORK_H
#define NETWORK_H

// We will have our Python script insert either
//     #define IS_GRU 0
// or  #define IS_GRU 1
// above these lines:

#define INPUT_SIZE      // Overwritten by python
#define LAYER1_SIZE     // Overwritten by python (used in Dense mode)
#define LAYER2_SIZE     // Overwritten by python (used in Dense mode)
#define LAYER3_SIZE     // Overwritten by python (used in Dense mode)

// For GRU mode, Python overwrites them as well:
// #define GRU1_UNITS
// #define GRU2_UNITS
// (and sets LAYER3_SIZE to the final output dimension)

//----------------------------------------------------
// Declarations
//----------------------------------------------------
void C_Network_Evaluate(float* inputs, float* outputs);

#if IS_GRU
void InitializeGRUStates(void);
#endif

// These exist for Dense networks
extern const float weights1[];
extern const float bias1[];
extern const float weights2[];
extern const float bias2[];
extern const float weights3[];
extern const float bias3[];

// These exist only for GRU networks (two GRU layers + final Dense).
// They will be empty if IS_GRU=0, but we declare them anyway:
extern const float gru1_kernel[];
extern const float gru1_recurrent_kernel[];
extern const float gru1_bias[];

extern const float gru2_kernel[];
extern const float gru2_recurrent_kernel[];
extern const float gru2_bias[];

// The final Dense layer after the second GRU reuses weights3, bias3.

#endif
