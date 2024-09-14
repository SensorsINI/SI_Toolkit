// main.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>      // Include for clock() function
#include "network.c"  // include the network implementation

#define NUM_RUNS 100000  // Define the number of times to run the network

int main(int argc, char *argv[]) {
    if (argc != INPUT_SIZE + 1) {
        printf("Please provide exactly %d input values\n", INPUT_SIZE);
        return 1;
    }

    // Parse input from command-line arguments
    float input[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = atof(argv[i + 1]);
    }

    float output[LAYER3_SIZE];  // output array


    // Start timing
    clock_t start_time = clock();  // Get initial clock ticks

    // Run the network evaluation NUM_RUNS times
    for (int i = 0; i < NUM_RUNS; i++) {
        C_Network_Evaluate(input, output);
    }

    // End timing
    clock_t end_time = clock();  // Get clock ticks after loop ends

    // Calculate time per call
    double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;  // Convert clock ticks to seconds
    double time_per_call = total_time*1000000.0 / NUM_RUNS;

    printf("Output of C network: ");
    for (int i = 0; i < LAYER3_SIZE; i++) {
        printf("%f ", output[i]);
    }
    printf("\n\n");

    // Print timing result
    printf("Total time for %d runs C implementation: %f seconds\n", NUM_RUNS, total_time);
    printf("Average time per call: %f us\n", time_per_call);

    return 0;
}
