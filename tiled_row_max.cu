__device__ void tile_rowmax(const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    // TODO: write row_max_out[r] = max over c of s_tile[r, c]
    for(int r = thread_id; r < tile_q; r += num_threads){
        float max = s_tile[r*tile_k];
        for(int c = 1; c < tile_k; c++){
            float current = s_tile[r*tile_k + c];
            if(current > max){
                max = current;
            }
        }
        row_max_out[r] = max;
    }
}
