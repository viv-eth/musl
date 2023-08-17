/*
 * Single-precision e^x function.
 *
 * Copyright (c) 2017-2018, Arm Limited.
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdint.h>
#include "libm.h"
#include "exp2f_data.h"

/*
EXP2F_TABLE_BITS = 5
EXP2F_POLY_ORDER = 3

ULP error: 0.502 (nearest rounding.)
Relative error: 1.69 * 2^-34 in [-ln2/64, ln2/64] (before rounding.)
Wrong count: 170635 (all nearest rounding wrong results with fma.)
Non-nearest ULP error: 1 (rounded ULP error)
*/

inline void snrt_fpu_fence() {
    unsigned tmp;
    __asm__ volatile(
        "fmv.x.w %0, fa0\n"
        "mv      %0, %0\n"
        : "+r"(tmp)::"memory");
}

uint64_t asuint64_safe(double f) {
    uint64_t result;
    snrt_fpu_fence();
	result = *(uint64_t *)&f;
    return result;
}

double asdouble_safe(uint64_t i) {
	double result;
	snrt_fpu_fence();
	result = *(double *)&i;
	return result;
}

#define N (1 << EXP2F_TABLE_BITS)
#define InvLn2N __exp2f_data.invln2_scaled
#define T __exp2f_data.tab
#define C __exp2f_data.poly_scaled

static inline uint32_t top12(float x)
{
	return asuint(x) >> 20;
}

float expf(float x)
{
	uint32_t abstop;
	uint64_t ki, t;
	double_t kd, xd, z, r, r2, y, sum0, sum1, s;
	// const double shift = 0x1.8p+52;

	xd = (double_t)x;
	abstop = top12(x) & 0x7ff;
	if (predict_false(abstop >= top12(88.0f))) {
		/* |x| >= 88 or x is nan.  */
		if (asuint(x) == asuint(-INFINITY))
			return 0.0f;
		if (abstop >= top12(INFINITY))
			return x + x;
		if (x > 0x1.62e42ep6f) /* x > log(0x1p128) ~= 88.72 */
			return __math_oflowf(0);
		if (x < -0x1.9fe368p6f) /* x < log(0x1p-150) ~= -103.97 */
			return __math_uflowf(0);
	}

	/* x*N/Ln2 = k + r with r in [-1/2, 1/2] and int k.  */
	z = InvLn2N * xd;

	/* Round and convert z to int, the result is in [-150*N, 128*N] and
	   ideally ties-to-even rule is used, otherwise the magnitude of r
	   can be bigger which gives larger approximation error.  */
#if TOINT_INTRINSICS
	kd = roundtoint(z);
	ki = converttoint(z);
#else
# define SHIFT __exp2f_data.shift
	kd = eval_as_double(z + SHIFT);
	// ki = asuint64(kd);
	ki = asuint64_safe(kd); // INFO: this fixes the RegWriteKnown issue
	kd -= SHIFT;
#endif
	r = z - kd;

	/* exp(x) = 2^(k/N) * 2^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1) */
	t = T[ki % N];
	t += ki << (52 - EXP2F_TABLE_BITS);
	// TODO: move t into integer register to avoid memory access/stack pointer accesses
	// try to take integer register value for conversion
	// TODO: RegWriteKnown add registers that are affected
	// s = asdouble(t);
	s = asdouble_safe(t);
	s = eval_as_float(s);
	// y = s * (C[2] * r + C[1] * r2 + C[0] * r + z * r2) = s * (r * (C[2] + C[0]) + r2 * (C[1] + z))
	//   = s * (sum0 + sum1)
	__asm__ volatile (
		"fmv.d   ft0, %[s]\n"
		"fmul.d  %[r2], %[r], %[r]\n"                 // r2 = r * r;
		"fadd.d  %[sum0], %[c2], %[c0]\n"         	  // sum0 = C[2] + C[0];
		"fmadd.d %[sum1], %[r], %[sum0], %[z]\n"      // sum1 = r * (C[2] + C[0]) + z;
		"fmadd.d %[y], %[r2], %[c1], %[sum1]\n"       // y = r2 * C[1] + y;
		"fmul.d  %[y], %[y], ft0\n"                   // y = y * s;
		: [y] "+f"(y), [sum0] "+f"(sum0), [sum1] "+f"(sum1)
		: [z] "f"(z), [r] "f"(r), [r2] "f"(r2), [s] "f"(s), [c0] "f"(C[0]), [c1] "f"(C[1]), [c2] "f"(C[2]), [temp_sum] "f"(0.0)
		: "ft0"
	);
	// z = C[0] * r + C[1];
	// r2 = r * r;
	// y = C[2] * r + 1;
	// y = z * r2 + y;
	// y = y  * s;
	return eval_as_float(y);
}
