#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// adjust this if you want more inputs, not including bias and class
#define N_INPUTS 2

// the number of epochs to run
#define EPOCHS 100000

// the learning parameter, usually 0.01
#define P 0.01

// bias (x0, must be 1), x1, x2, ..., class (must be 1 or -1)
#define DATA_COLS (N_INPUTS + 2)
static double data[][DATA_COLS] = {
        {1, 1, 4, -1},
        {1, 2, 9, 1},
        {1, 5, 6, 1},
        {1, 4, 5, 1},
        {1, 6, 0.7, -1},
        {1, 1, 1.5, -1},
};
static const size_t n_rows = (size_t) (sizeof (data) / sizeof (data)[0]);

#define N_WEIGHTS (N_INPUTS + 1)
static double weights[N_WEIGHTS];

void pass() {
    for (size_t row_idx = 0; row_idx < n_rows; row_idx++) {
        double *sample = data[row_idx];

        double weight_sum = 0;
        for (size_t weight_idx = 0; weight_idx < N_WEIGHTS; weight_idx++) {
            weight_sum += weights[weight_idx] * sample[weight_idx];
        }

        // the real class is the last column of data
        int real_class = (int) sample[DATA_COLS - 1];

        // update weights
        for (size_t weight_idx = 0; weight_idx < N_WEIGHTS; weight_idx++) {
            weights[weight_idx] += sample[weight_idx] * P * (real_class - weight_sum);
        }
    }
}

int main(void) {
    srand(time(NULL));

    // randomise weights
    for (size_t weight_idx = 0; weight_idx < N_WEIGHTS; weight_idx++) {
        weights[weight_idx] = (double) rand() / (double) RAND_MAX;
    }

    puts("Starting weights:");
    for (size_t weight_idx = 0; weight_idx < N_WEIGHTS; weight_idx++) {
        printf("w%zu = %f\n", weight_idx, weights[weight_idx]);
    }

    for (size_t epoch_idx = 0; epoch_idx < EPOCHS; epoch_idx++) {
        pass();
    }

    puts("Resulting weights:");

    for (size_t weight_idx = 0; weight_idx < N_WEIGHTS; weight_idx++) {
        printf("w%zu = %f\n", weight_idx, weights[weight_idx]);
    }

    return 0;
}
