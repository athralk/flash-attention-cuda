void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {
    // TODO: allocate , launch qk_scores -> softmax_rows -> pv_matmul, free 
    float* d_scores = nullptr;
    cudaMalloc(&d_scores, seq_len*seq_len*sizeof(float));

    dim3 block(16,16);
    dim3 threads_qk ((seq_len + block.x - 1)/ block.x, (seq_len + block.y - 1) / block.y);

    qk_scores <<<threads_qk, block>>> (d_q, d_k, d_scores, seq_len, head_dim);

    int block1d = 256;
    int threads_sf = (seq_len + block1d - 1) / block1d;
    softmax_rows <<<threads_sf, block1d>>> (d_scores, seq_len, seq_len);

    dim3 threads_pv ((head_dim + block.x - 1)/ block.x, (seq_len + block.y - 1) / block.y);
    pv_matmul <<<threads_pv, block>>> (d_scores, d_v, d_out, seq_len, head_dim);

    cudaFree(d_scores);
}
