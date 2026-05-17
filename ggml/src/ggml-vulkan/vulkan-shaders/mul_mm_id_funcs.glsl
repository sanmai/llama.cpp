#ifdef MUL_MAT_ID
shared uvec2 row_ids[BN];
uint _ne1;

void load_row_ids(uint expert_idx, uint ic) {
    const uint row_base = data_expert_offset[expert_idx];
    _ne1 = data_expert_offset[expert_idx + 1] - row_base;

    for (uint i = gl_LocalInvocationIndex; i < BN && ic * BN + i < _ne1; i += BLOCK_SIZE) {
        row_ids[i] = data_row_ids[row_base + ic * BN + i];
    }
    barrier();
}
#endif // MUL_MAT_ID
