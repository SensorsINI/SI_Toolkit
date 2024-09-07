#ifndef NETWORK_H
#define NETWORK_H

// Just a template, the values are adjusted with python script
#define INPUT_SIZE      // Input size (as seen from your debugger, you can adjust as needed)
#define LAYER1_SIZE     // First Dense layer size
#define LAYER2_SIZE     // Second Dense layer size
#define LAYER3_SIZE     // Third Dense layer size (new for the third Dense layer)

// Declare the function and global variables
void C_Network_Evaluate(float* inputs, float* outputs);

// Other necessary declarations if required
extern const float weights1[];
extern const float bias1[];
extern const float weights2[];
extern const float bias2[];
extern const float weights3[];
extern const float bias3[];

#endif