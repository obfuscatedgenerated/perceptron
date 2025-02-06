#include <stdio.h>

// adjust this if you want more inputs
#define DATA_COLS 4

// bias (x0, must be 1), x1, x2, ..., class (must be 1 or -1)
static double data[][DATA_COLS] = {
    {1, 1, 4, -1},
    {1, 2, 9, 1},
    {1, 5, 6, 1},
    {1, 4, 5, 1},
    {1, 6, 0.7, -1},
    {1, 1, 1.5, -1},
};
static const size_t n_rows = (size_t) (sizeof (data) / sizeof (data)[0]);

static double weights[DATA_COLS - 1];

int pass() {
    int made_change = 0;

    for (size_t row_idx = 0; row_idx < n_rows; row_idx++) {
        double *sample = data[row_idx];

        double weight_sum = 0;
        for (size_t weight_idx = 0; weight_idx < (DATA_COLS - 1); weight_idx++) {
            weight_sum += weights[weight_idx] * sample[weight_idx];
        }

        // get the sign of the sum to get the predicted class, the real class is the last column of data
        int predicted_class = (weight_sum > 0) - (weight_sum < 0);
        int real_class = (int) sample[DATA_COLS - 1];

        if (predicted_class != real_class) {
            made_change = 1;

            for (size_t weight_idx = 0; weight_idx < (DATA_COLS - 1); weight_idx++) {
                weights[weight_idx] += real_class * sample[weight_idx];
            }
        }
    }

    return made_change;
}

int main(void) {
    // epoch_num will effectively start at 1 (so i haven't named it clock_idx)
    size_t epoch_num = 0;
    int loop = 1;
    while (loop) {
        epoch_num++;
        loop = pass();
    }

    printf("Finished fit in %zu epochs\nResulting weights:\n", epoch_num - 1);

    for (size_t weight_idx = 0; weight_idx < (DATA_COLS - 1); weight_idx++) {
        printf("w%zu = %f\n", weight_idx, weights[weight_idx]);
    }

    return 0;
}
