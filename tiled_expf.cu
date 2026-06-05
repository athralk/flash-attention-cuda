__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    // TODO: for each (r, c) in the tile, set s_tile[r*tile_k+c] = expf(s_tile[r*tile_k+c] - row_max[r])

    for(int r = thread_id; r < tile_q; r += num_threads){
        // first compute the maximum
        /*float max = s_tile[r*tile_k];
        for(int c = 1; c < tile_k; c++){
            float current = s_tile[r*tile_k + c];
            if(current > max){
                max = current;
            }
        }
        
        __syncthreads();
        */
        // now do expf()
        for(int x = 0; x < tile_k; x++){
            s_tile[r*tile_k + x] = expf(s_tile[r*tile_k + x] - row_max[r]);
        }
    }
}

//re wrote row max cus i didnt know we were using const float* row_max from the prev device helper ie tiled_row_max.cu
