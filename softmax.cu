__global__ void softmax_rows(float* matrix, int rows, int cols) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < rows) {
        int row_offset = row * cols;

        float max_val = matrix[row_offset];
        for (int i = 1; i < cols; i++) {
            float val = matrix[row_offset + i];
            if (val > max_val) {
                max_val = val;
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum += expf(matrix[row_offset + i] - max_val);
        }

        for (int j = 0; j < cols; j++) {
            matrix[row_offset + j] = expf(matrix[row_offset + j] - max_val) / sum;
        }
    }
}