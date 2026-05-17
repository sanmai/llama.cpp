#ifdef MUL_MAT_ID
shared u16vec2 row_ids[BN];
uint _ne1;

void load_row_ids(uint expert_idx, uint ic) {
    const uint row_base = data_expert_offset[expert_idx];
    _ne1 = data_expert_offset[expert_idx + 1] - row_base;

    for (uint i = gl_LocalInvocationIndex; i < BN && ic * BN + i < _ne1; i += BLOCK_SIZE) {
        const uint row_id = data_row_ids[row_base + ic * BN + i];
        // i00 in the low 16 bits, i01 (token index, capped at 65535) in the high 16
        row_ids[i] = u16vec2(row_id & 0xffff, row_id >> 16);
    }
    barrier();
}
#endif // MUL_MAT_ID
