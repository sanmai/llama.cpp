#ifdef MUL_MAT_ID
shared u16vec2 row_ids[BN];
uint _ne1;

void load_row_ids(uint expert_idx, uint ic) {
    _ne1 = data_expert_count[expert_idx];

    const uint row_base = expert_idx * p.nei1 * p.nei0 + ic * BN;

    for (uint i = gl_LocalInvocationIndex; i < BN && ic * BN + i < _ne1; i += BLOCK_SIZE) {
        const uint row_id = data_row_ids[row_base + i];
        row_ids[i] = u16vec2(row_id & 0xffff, row_id >> 16);
    }
    barrier();
}
#endif // MUL_MAT_ID
