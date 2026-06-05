__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    // TODO: cooperatively fill row_sum_out[r] with the sum of p_tile row r
    for(int r = thread_id; r < tile_q; r += num_threads){
        float sum = 0.0f;
        for(int c = 0; c < tile_k; c++){
            sum += p_tile[r*tile_k + c];
        }      
        row_sum_out[r] = sum;    
    }
}

//this is O(n^2) and to break it down to O(log(N)) it will require binary tree 
//and __shf__shfl_down_sync(0xffffffff) knowledge?
