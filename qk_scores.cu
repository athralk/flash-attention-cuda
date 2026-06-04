__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    
    int col = threadIdx.x + blockIdx.x * blockDim.x; 
    int row = threadIdx.y + blockIdx.y * blockDim.y; 

    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        
        for (int i = 0; i < head_dim; i++) {
            float q_val = q[row * head_dim + i]; 
            float k_val = k[col * head_dim + i]; 
            sum += q_val * k_val;
        }
        
        float scale = 1.0f / sqrtf((float)head_dim);
        scores[row * seq_len + col] = sum * scale;
    }
}