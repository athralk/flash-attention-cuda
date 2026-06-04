__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    //out[i,d] = sum of p[i, j] * v[j,d];
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < seq_len && col < head_dim){
        float sum = 0.0f;
        for(int j = 0; j < seq_len; j++){
            sum += p[row*seq_len + j] * v[j*head_dim + col];
        }
        out[row*head_dim + col] = sum;
    }
}
