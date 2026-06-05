__device__ void accumulate_pv(const float* p_tile, const float* v_tile, float* out_acc, int tile_q, int tile_k, int head_dim, int thread_id, int num_threads) {
    // TODO: cooperatively add P_tile * V_tile into out_acc
    int size = head_dim * tile_q;
    for(int linear_idx = thread_id; linear_idx < size; linear_idx += num_threads){
        int i = linear_idx / head_dim;
        int j = linear_idx % head_dim;
        float sum = 0.0f;

        for(int x = 0; x < tile_k; x++){
            sum += p_tile[i * tile_k + x] * v_tile[x * head_dim + j]; 
        }
        out_acc[linear_idx] += sum;
    }
}

//in this since tile_k is the vanishable part we do the row and col using head_dim since it is the output part
//and the tile_k is used to loop the x axis

