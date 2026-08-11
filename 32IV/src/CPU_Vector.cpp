#include <cstdint>
#include <algorithm>
#include "CPU.h"
#include "CPU_Vector.h"

void CPU::vector_decode() {
    vectorInstruction.funct6 = (instruction >> 26) & 0x3F;
    vectorInstruction.funct3 = (instruction >> 12) & 0x07;
    vectorInstruction.vd = (instruction >> 7) & 0x1F;
    vectorInstruction.rd = (instruction >> 7) & 0x1F;
    vectorInstruction.vs1 = (instruction >> 15) & 0x1F;
    vectorInstruction.rs1 = (instruction >> 15) & 0x1F;
    vectorInstruction.vs2 = (instruction >> 20) & 0x1F;
    vectorInstruction.rs2 = (instruction >> 20) & 0x1F;
    vectorInstruction.vm = (instruction >> 25) & 0x01;
    vectorInstruction.imm_Z_VSETVLI = (instruction >> 20) & 0x7FF;
    vectorInstruction.imm_Z_VSETIVLI = (instruction >> 20) & 0x3FF;
    vectorInstruction.imm_U = (instruction >> 15) & 0x1F;
    vectorInstruction.bit_31 = (instruction >> 31) & 0x01;
    vectorInstruction.bit_30 = (instruction >> 30) & 0x01;
    // Control signals (reg_write_enable / mem_*_enable / vector_mem_enable) are set by the
    // per-opcode switch in CPU::decode() - this function only decodes instruction fields.

    uint8_t opcode = instruction & 0x7F;
    vectorInstruction.ls_supported = false;
    vectorInstruction.ls_whole_register = false;
    vectorInstruction.ls_fault_only_first = false;
    vectorInstruction.eew_bytes = 0;

    if (opcode == 0x07 || opcode == 0x27) {
        // Load/store-only fields: bits 31:29=nf, 28=mew, 27:26=mop (same bit range as
        // funct6, decoded separately here for clarity in this context).
        vectorInstruction.nf  = (instruction >> 29) & 0x07;
        vectorInstruction.mew = (instruction >> 28) & 0x01;
        vectorInstruction.mop = (instruction >> 26) & 0x03;

        // `width` (=funct3) selects EEW: 000=8b, 101=16b, 110=32b, 111=64b. Other values
        // (and mew=1, used for >64-bit elements) are reserved/unsupported for now. For
        // indexed instructions this is the INDEX element width, not the data width (data
        // uses SEW from vtype instead) - see vector_load_store_execute()/vector_read().
        switch (vectorInstruction.funct3) {
            case 0x0: vectorInstruction.eew_bytes = 1; break;
            case 0x5: vectorInstruction.eew_bytes = 2; break;
            case 0x6: vectorInstruction.eew_bytes = 4; break;
            case 0x7: vectorInstruction.eew_bytes = 8; break;
            default:  vectorInstruction.eew_bytes = 0; break;
        }

        // mop: 00=unit-stride, 01=indexed-unordered, 10=strided, 11=indexed-ordered.
        bool unit_stride = (vectorInstruction.mop == 0x0);
        bool strided      = (vectorInstruction.mop == 0x2);
        bool indexed      = (vectorInstruction.mop == 0x1 || vectorInstruction.mop == 0x3);

        // For unit-stride only, `vs2` (bits 24:20) doubles as lumop/sumop, selecting
        // between the regular element load/store and the whole-register / fault-only-
        // first sub-modes (mask load/store, sub-mode 0b01011, is still not implemented).
        // For strided it's `rs2` (the stride GPR); for indexed it's the index vector
        // register - both already decoded above as vs2/rs2 (same raw bits).
        uint8_t lsumop = vectorInstruction.vs2;
        bool regular_sub_mode = unit_stride && (lsumop == 0x00);
        vectorInstruction.ls_whole_register   = unit_stride && (lsumop == 0x08);
        vectorInstruction.ls_fault_only_first = unit_stride && (opcode == 0x07) && (lsumop == 0x10);

        uint32_t nfields = vectorInstruction.nf + 1u;
        bool nfields_in_range = (vectorInstruction.vd + nfields) <= 32;

        if (vectorInstruction.ls_whole_register) {
            // Whole-register load/store ignores vl/vtype/masking entirely; only
            // power-of-2 NFIELDS with an aligned vd are legal (spec: "the encoded number
            // of registers must be a power of 2 and the vector register numbers must be
            // aligned as with a vector register group").
            bool nfields_pow2 = (nfields == 1 || nfields == 2 || nfields == 4 || nfields == 8);
            bool vd_aligned = (vectorInstruction.vd % nfields) == 0;
            vectorInstruction.ls_supported =
                !vectorInstruction.mew && vectorInstruction.eew_bytes != 0 &&
                nfields_pow2 && vd_aligned && nfields_in_range;
        } else {
            // Regular elements, fault-only-first, and segments (nf>0) for any of the
            // three addressing modes. Anything else (reserved lumop/sumop sub-modes like
            // mask load/store, mew=1, reserved width) is a safe no-op, mirroring the
            // scalar decoder's silent "unimplemented -> NOP" default case.
            vectorInstruction.ls_supported =
                !vectorInstruction.mew && vectorInstruction.eew_bytes != 0 && nfields_in_range &&
                (regular_sub_mode || vectorInstruction.ls_fault_only_first || strided || indexed);
        }
    }
}

void CPU::vector_config_execute() {
    enum VectorConfigType {
        VSETVLI,
        VSETIVLI,
        VSETVL 
    };
    
    VectorConfigType configType;
    uint32_t proposed_vtype;

    if (!vectorInstruction.bit_31) {
        configType = VSETVLI;
        proposed_vtype = vectorInstruction.imm_Z_VSETVLI;
    } 
    else if (vectorInstruction.bit_30) { 
        configType = VSETIVLI;
        proposed_vtype = vectorInstruction.imm_Z_VSETIVLI;
    } 
    else {
        configType = VSETVL;
        proposed_vtype = registers[vectorInstruction.rs2]; 
    }

    uint8_t vtype_sew = (proposed_vtype >> 3) & 0x7;
    uint8_t vtype_lmul = proposed_vtype & 0x7;
    
    if (vtype_sew > 0x3 || vtype_lmul == 0x4) {
        vtype = 0x80000000; 
        vl = 0;
        aluResult = 0; 
        return; 
    }
    
    float lmul;
    uint32_t sew = 8 * (1 << vtype_sew);
    
    switch(vtype_lmul) {
        case 0x5: lmul = 0.125f; break;
        case 0x6: lmul = 0.25f;  break;
        case 0x7: lmul = 0.5f;   break;  
        default:  lmul = static_cast<float>(1 << vtype_lmul); break;
    }
    
    uint32_t vlmax = static_cast<uint32_t>((VLEN * lmul) / sew);

    uint32_t avl;
    if (configType == VSETIVLI) {
        avl = vectorInstruction.imm_U; 
    } 
    else {
        if (vectorInstruction.rs1 == 0x0) {
            avl = vlmax; 
        } else {
            avl = registers[vectorInstruction.rs1];
        }
    }

    uint32_t new_vl;
    if (avl <= vlmax) {
        new_vl = avl;
    } else {
        new_vl = vlmax;
    }

    vtype = proposed_vtype; 
    vl = new_vl; 
    aluResult = new_vl; 
}
// Address generation: for each lane (segment, when nf>0), compute its base memory address
// and whether it's active (in-range of vl, and unmasked or set in the v0 mask register -
// masking applies at whole-segment granularity per spec). Parallel across elements in
// spirit - a real HW unit would compute all lanes combinationally in one cycle.
void CPU::vector_load_store_execute() {
    vec_lane_active.fill(false);
    vec_lane_addr.fill(0);
    vec_num_lanes = 0;

    if (!vectorInstruction.ls_supported || vectorInstruction.ls_whole_register) {
        // Unsupported encoding: no address generated, safe no-op downstream. Whole-
        // register transfers don't need per-lane address-gen at all - they're a single
        // flat contiguous block, computed directly in vector_read().
        return;
    }

    // Indexed instructions use SEW (current vtype) as the data element width; unit-stride
    // and strided use the instruction's own encoded EEW.
    bool indexed = (vectorInstruction.mop == 0x1 || vectorInstruction.mop == 0x3);
    uint32_t data_eew_bytes = indexed
        ? (1u << ((vtype >> 3) & 0x7))
        : vectorInstruction.eew_bytes;
    uint32_t index_eew_bytes = vectorInstruction.eew_bytes; // only meaningful when indexed

    uint32_t base_addr = registers[vectorInstruction.rs1];
    uint32_t nfields = vectorInstruction.nf + 1u;
    uint32_t lanes = VLEN_BYTES / data_eew_bytes; // vlmax for LMUL<=1 at this data EEW

    vec_num_lanes = lanes;
    for (uint32_t i = 0; i < lanes; i++) {
        uint32_t segment_addr;
        if (vectorInstruction.mop == 0x2) {
            // Strided: consecutive segments are `rs2` bytes apart - rs2 holds a signed
            // byte stride, so negative and zero strides (spec-legal) fall out naturally.
            int64_t stride = (int64_t)(int32_t)registers[vectorInstruction.rs2];
            segment_addr = (uint32_t)((int64_t)base_addr + (int64_t)i * stride);
        } else if (indexed) {
            // Indexed: segment i's address is base + offset[i], where offset[i] is read
            // from the index vector vs2 at index_eew_bytes granularity, zero-extended (or
            // truncated to the low 32 bits if index_eew_bytes > 4, per spec).
            uint32_t offset = 0;
            if (i < VLEN_BYTES / index_eew_bytes) {
                for (uint32_t b = 0; b < index_eew_bytes && b < 4; b++) {
                    offset |= (uint32_t)vregfile.vregs[vectorInstruction.vs2][i * index_eew_bytes + b] << (8 * b);
                }
            }
            segment_addr = base_addr + offset;
        } else {
            // Unit-stride: consecutive segments are packed contiguously, nfields*eew apart.
            segment_addr = base_addr + i * (nfields * data_eew_bytes);
        }
        vec_lane_addr[i] = segment_addr;

        bool in_range = (i < vl);
        bool mask_bit = vectorInstruction.vm
            ? true
            : ((vregfile.vregs[0][i / 8] >> (i % 8)) & 0x1);
        vec_lane_active[i] = in_range && mask_bit;
    }
}

// Per-lane (per-segment) memory access: lanes are no longer guaranteed contiguous once
// strided/indexed addressing is in play, so each lane does its own access against its own
// vec_lane_addr[i] - still "parallel across elements" in spirit (one cycle overall), just
// no longer a single bulk read/write like the original unit-stride-only version assumed.
// Each segment carries `nfields` fields, field f landing in register vd+f.
void CPU::vector_read() {
    if (!vectorInstruction.ls_supported) {
        return;
    }
    uint8_t opcode = instruction & 0x7F;

    if (vectorInstruction.ls_whole_register) {
        // Whole-register transfers ignore vl/vtype/masking: NFIELDS consecutive VLEN_BYTES
        // blocks, moved as one flat contiguous region between memory and vd..vd+NFIELDS-1.
        uint32_t nfields = vectorInstruction.nf + 1u;
        uint32_t base_addr = registers[vectorInstruction.rs1];
        uint32_t total_bytes = nfields * VLEN_BYTES;

        if (opcode == 0x07) {
            std::vector<uint8_t> data = memory->read_bytes(base_addr, total_bytes);
            for (uint32_t f = 0; f < nfields; f++) {
                for (uint32_t b = 0; b < VLEN_BYTES; b++) {
                    vec_load_buffer[f][b] = data[f * VLEN_BYTES + b];
                }
            }
        } else if (opcode == 0x27) {
            std::vector<uint8_t> data(total_bytes);
            for (uint32_t f = 0; f < nfields; f++) {
                for (uint32_t b = 0; b < VLEN_BYTES; b++) {
                    data[f * VLEN_BYTES + b] = vregfile.vregs[vectorInstruction.vd + f][b];
                }
            }
            memory->write_bytes(base_addr, data, std::vector<bool>(total_bytes, true));
        }
        return;
    }

    bool indexed = (vectorInstruction.mop == 0x1 || vectorInstruction.mop == 0x3);
    uint32_t data_eew_bytes = indexed
        ? (1u << ((vtype >> 3) & 0x7))
        : vectorInstruction.eew_bytes;
    uint32_t nfields = vectorInstruction.nf + 1u;

    if (opcode == 0x07) { // Load
        bool fof_trimmed = false; // set once a fault-only-first probe fails on segment i>0
        for (uint32_t i = 0; i < vec_num_lanes; i++) {
            bool active = vec_lane_active[i] && !fof_trimmed;

            if (active && vectorInstruction.ls_fault_only_first) {
                // Probe every field of this segment before committing to a real read.
                bool all_valid = true;
                for (uint32_t f = 0; f < nfields && all_valid; f++) {
                    all_valid = memory->probe(vec_lane_addr[i] + f * data_eew_bytes, data_eew_bytes);
                }
                if (!all_valid) {
                    // Spec: only segment 0 can take a synchronous trap (vl left
                    // unmodified); this simulator has no trap/exception mechanism
                    // anywhere (the scalar core doesn't either), so the closest faithful
                    // approximation is to abandon the load with vl untouched and nothing
                    // written, rather than fabricate data. Segment i>0 instead trims vl
                    // to i, per spec, and every remaining segment is treated as past vl.
                    if (i == 0) {
                        return;
                    }
                    vl = i;
                    fof_trimmed = true;
                    active = false;
                }
            }

            uint32_t byte_base = i * data_eew_bytes;
            for (uint32_t f = 0; f < nfields; f++) {
                if (active) {
                    std::vector<uint8_t> elem = memory->read_bytes(vec_lane_addr[i] + f * data_eew_bytes, data_eew_bytes);
                    for (uint32_t b = 0; b < data_eew_bytes; b++) {
                        vec_load_buffer[f][byte_base + b] = elem[b];
                    }
                } else {
                    // Inactive lanes (masked out, past vl, or just fault-only-first
                    // trimmed) keep their old register value - this one rule implements
                    // both "undisturbed" and "agnostic" policy the same way, a valid
                    // choice for an agnostic policy that keeps behavior deterministic.
                    for (uint32_t b = 0; b < data_eew_bytes; b++) {
                        vec_load_buffer[f][byte_base + b] = vregfile.vregs[vectorInstruction.vd + f][byte_base + b];
                    }
                }
            }
        }
    } else if (opcode == 0x27) { // Store
        for (uint32_t i = 0; i < vec_num_lanes; i++) {
            if (!vec_lane_active[i]) continue; // masked-out/tail segment: no memory access at all

            uint32_t byte_base = i * data_eew_bytes;
            for (uint32_t f = 0; f < nfields; f++) {
                std::vector<uint8_t> elem(data_eew_bytes);
                for (uint32_t b = 0; b < data_eew_bytes; b++) {
                    elem[b] = vregfile.vregs[vectorInstruction.vd + f][byte_base + b]; // vd doubles as vs3
                }
                memory->write_bytes(vec_lane_addr[i] + f * data_eew_bytes, elem, std::vector<bool>(data_eew_bytes, true));
            }
        }
    }
}

// Single commit point for every vector opcode (config/load/store), dispatched by opcode
// so the same signal (vector_instruction_active) can drive all of them uniformly from
// CPU::writeback() instead of splitting config off into the generic scalar path.
void CPU::vector_writeback() {
    uint8_t opcode = instruction & 0x7F;
    switch (opcode) {
        case 0x57: // Config (VSETVLI/VSETIVLI/VSETVL): commit the negotiated vl to rd,
                   // same behavior as the original scalar-only vector_writeback().
            if (vectorInstruction.rd != 0x0) {
                registers[vectorInstruction.rd] = aluResult;
            }
            break;

        case 0x07: { // Load: commit the merged buffer into the vector register file
            if (!vectorInstruction.ls_supported) break;
            uint32_t nfields = vectorInstruction.nf + 1u;

            if (vectorInstruction.ls_whole_register) {
                for (uint32_t f = 0; f < nfields; f++) {
                    vregfile.vregs[vectorInstruction.vd + f] = vec_load_buffer[f];
                }
                break;
            }

            bool indexed = (vectorInstruction.mop == 0x1 || vectorInstruction.mop == 0x3);
            uint32_t data_eew_bytes = indexed
                ? (1u << ((vtype >> 3) & 0x7))
                : vectorInstruction.eew_bytes;
            uint32_t total_bytes = vec_num_lanes * data_eew_bytes;
            for (uint32_t f = 0; f < nfields; f++) {
                for (uint32_t b = 0; b < total_bytes; b++) {
                    vregfile.vregs[vectorInstruction.vd + f][b] = vec_load_buffer[f][b];
                }
            }
            break;
        }

        case 0x27: // Store: nothing to commit, already fully done in vector_read()
        default:
            break;
    }
}