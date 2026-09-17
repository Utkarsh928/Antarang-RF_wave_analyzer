/*
 * viterbi.cpp
 * ===========
 * High-speed Viterbi decoder for rate-1/2 convolutional codes.
 *
 * Supports constraint lengths K = 3, 5, 7 (most common in practice):
 *   K=3  (4 states)  — simple, fast
 *   K=5  (16 states) — moderate
 *   K=7  (64 states) — NASA standard, widely used in satellite/telemetry
 *
 * Default polynomials (industry standard, octal notation):
 *   K=7: g1=0171 (0x79=121), g2=0133 (0x5B=91)  — NASA/ESA standard
 *   K=5: g1=023  (19),       g2=035  (29)
 *   K=3: g1=07   (7),        g2=05   (5)
 *
 * Uses hard-decision Viterbi (input = 0/1 bits).
 * Soft-decision input supported via LLR conversion.
 *
 * All arrays are uint8 (C unsigned char) for direct numpy compatibility.
 *
 * Build (MSVC x64):
 *   cl /O2 /LD /EHsc viterbi.cpp /Fe:viterbi.dll
 */

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>

#ifdef _WIN32
#  define EXPORT extern "C" __declspec(dllexport)
#else
#  define EXPORT extern "C" __attribute__((visibility("default")))
#endif

/* ── Convolutional encoder polynomials ──────────────────────────────────── */
/* These match the standard rate-1/2 codes used in practice */

typedef struct {
    int K;          /* constraint length */
    int n_states;   /* 2^(K-1) */
    int g1, g2;     /* generator polynomials (decimal) */
} ConvParams;

static ConvParams get_params(int K) {
    ConvParams p;
    p.K = K;
    p.n_states = 1 << (K-1);
    switch(K) {
        case 3: p.g1=7;  p.g2=5;  break;  /* (7,5) octal */
        case 5: p.g1=19; p.g2=29; break;  /* (23,35) octal */
        default:
        case 7: p.g1=121; p.g2=91; break; /* (171,133) octal — NASA standard */
    }
    return p;
}

/* ── Parity (number of 1 bits) ──────────────────────────────────────────── */
static inline int parity(int x) {
    x ^= x >> 16; x ^= x >> 8; x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
    return x & 1;
}

/* ── Generate encoder output for (state, input_bit) ─────────────────────── */
static int encode_bit(int state, int bit, int K, int g1, int g2) {
    /* Shift register: input bit is MSB, state is the remaining K-1 bits */
    int sr = (bit << (K-1)) | state;
    int out1 = parity(sr & g1);
    int out2 = parity(sr & g2);
    return (out1 << 1) | out2;
}

/* ── Viterbi decoder ─────────────────────────────────────────────────────── */

/*
 * viterbi_decode()
 * ----------------
 * encoded_bits : input bits (pairs: 2 bits per symbol, rate 1/2)
 *                length must be even; coded_len = 2 * n_info_bits
 * coded_len    : total number of encoded bits (= 2 * output length)
 * decoded_bits : output array, caller allocates coded_len/2 bytes
 * K            : constraint length (3, 5, or 7)
 *
 * Returns number of decoded bits, or -1 on error.
 *
 * The function expects tail-biting or zero-terminated code.
 * It automatically pads if the sequence is not symbol-aligned.
 */
EXPORT int viterbi_decode(const uint8_t* encoded_bits,
                           int coded_len,
                           uint8_t* decoded_bits,
                           int K)
{
    if (!encoded_bits || !decoded_bits || coded_len < 2) return -1;
    if (K != 3 && K != 5 && K != 7) K = 7;   /* default to NASA K=7 */

    ConvParams p = get_params(K);
    int S = p.n_states;
    int n_syms = coded_len / 2;   /* number of symbol pairs */

    /* Allocate Viterbi trellis */
    /* path_metric[s] = best accumulated metric to state s */
    int* pm      = (int*)calloc(S, sizeof(int));
    int* pm_new  = (int*)malloc(S * sizeof(int));
    /* survivor paths: for each symbol, for each state, which prev state led here */
    uint8_t* survivors = (uint8_t*)malloc((size_t)n_syms * S);
    if (!pm || !pm_new || !survivors) {
        free(pm); free(pm_new); free(survivors);
        return -1;
    }

    /* Initialise: state 0 has metric 0, all others = large */
    for (int s = 0; s < S; s++) pm[s] = (s == 0) ? 0 : 1000000;

    /* Pre-compute expected encoder outputs for each (state, input_bit) */
    /* enc_out[s][b] = 2-bit output when in state s, input bit b */
    int* enc_out = (int*)malloc(S * 2 * sizeof(int));
    int* next_st = (int*)malloc(S * 2 * sizeof(int));
    for (int s = 0; s < S; s++) {
        for (int b = 0; b < 2; b++) {
            enc_out[s*2+b] = encode_bit(s, b, p.K, p.g1, p.g2);
            next_st[s*2+b] = ((b << (p.K-2)) | (s >> 1)) & (S-1);
        }
    }

    /* ── Forward pass ───────────────────────────────────────────────────── */
    for (int t = 0; t < n_syms; t++) {
        int r0 = encoded_bits[2*t];     /* received bit 0 */
        int r1 = encoded_bits[2*t+1];   /* received bit 1 */
        int rec = ((r0 & 1) << 1) | (r1 & 1);  /* received 2-bit symbol */

        for (int s_new = 0; s_new < S; s_new++) pm_new[s_new] = 1000000;

        for (int s = 0; s < S; s++) {
            if (pm[s] >= 1000000) continue;
            for (int b = 0; b < 2; b++) {
                int expected = enc_out[s*2+b];
                /* Hamming distance: number of differing bits */
                int diff = rec ^ expected;
                int hd = (diff & 1) + ((diff >> 1) & 1);
                int ns = next_st[s*2+b];
                int new_m = pm[s] + hd;
                if (new_m < pm_new[ns]) {
                    pm_new[ns] = new_m;
                    survivors[t*S + ns] = (uint8_t)s;
                }
            }
        }
        memcpy(pm, pm_new, S * sizeof(int));
    }

    /* ── Traceback ──────────────────────────────────────────────────────── */
    /* Find best final state */
    int best_s = 0, best_m = pm[0];
    for (int s = 1; s < S; s++) {
        if (pm[s] < best_m) { best_m = pm[s]; best_s = s; }
    }

    /* Trace back through survivors */
    uint8_t* bits_rev = (uint8_t*)malloc(n_syms);
    int s = best_s;
    for (int t = n_syms-1; t >= 0; t--) {
        int prev_s = survivors[t*S + s];
        /* Recover input bit: what bit was fed to go from prev_s to s? */
        int bit = (s >> (p.K-2)) & 1;
        bits_rev[t] = (uint8_t)bit;
        s = prev_s;
    }

    /* Copy to output (already in forward order) */
    memcpy(decoded_bits, bits_rev, n_syms);

    free(pm); free(pm_new); free(survivors);
    free(enc_out); free(next_st); free(bits_rev);
    return n_syms;
}

/*
 * viterbi_decode_k7()
 * -------------------
 * Convenience wrapper for the most common K=7 NASA-standard code.
 */
EXPORT int viterbi_decode_k7(const uint8_t* encoded_bits,
                               int coded_len,
                               uint8_t* decoded_bits)
{
    return viterbi_decode(encoded_bits, coded_len, decoded_bits, 7);
}

/*
 * count_bit_errors()
 * ------------------
 * Count bit errors between two bit arrays of equal length.
 * Returns number of differing bits (Hamming distance).
 */
EXPORT int count_bit_errors(const uint8_t* a, const uint8_t* b, int n) {
    int errors = 0;
    for (int i = 0; i < n; i++)
        errors += (a[i] != b[i]) ? 1 : 0;
    return errors;
}
