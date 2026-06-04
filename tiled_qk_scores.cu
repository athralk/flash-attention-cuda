__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    
    int size_score = tile_k * tile_q;
    
    for (int linear_idx = thread_id; linear_idx < size_score; linear_idx += num_threads) {
        
        int i = linear_idx / tile_k;
        int j = linear_idx % tile_k; 

        float sum = 0.0f;

        for (int x = 0; x < head_dim; x++) {
            sum += q_tile[i * head_dim + x] * k_tile[j * head_dim + x];
        }
        
        s_tile[linear_idx] = sum * scale;        
    }
}