# ADR-079 P4.32a: the short-vector harness every ARMv6 F32 kernel generator (scripts/gen_armv6_*) builds on.
#
# With FPSCR LEN = 4, STRIDE = 1 the VFP11 turns a data-processing op whose destination is outside bank 0
# (s8-s31) into a vector of four; a bank-0 Sm stays a scalar operand (bench44 `vfpvec`). The mode is
# per thread and every piece of compiled code assumes LEN = 0 (AAPCS), so a kernel holds it only inside one
# generated asm function, around loops that are integer code, and calls nothing while it is set:
#
#   enter(e, tmp)   vmrs/bic/orr/vmsr: LEN = 4, STRIDE = 1. RunFast (FZ | DN) is the thread's (ADR-078).
#   leave(e, tmp)   clears LEN and STRIDE only. Restoring a saved FPSCR instead would also roll back
#                   the cumulative exception flags the kernel raised (scripts/armv6_fpflags.c reads them).
#   lint(lines)     the checks below, on a generator's whole output; generators call it before writing.
#
# Inside a region lint rejects any way out of the function (bl/blx/bx/svc, a pc write, a ldm/pop of pc),
# a second vmsr, double-precision data processing (its vectors would run on the d banks), and a vector op
# whose destination or Sn/Sm vector would wrap around its bank (s14 at LEN 4 is s14, s15, s8, s9 -- legal,
# never intended). Every function must leave the region before it returns. Signals and context switches
# are the kernel's to get right; scripts/armv6_shortvec_gate.c is the evidence that they do (gate 1).
import re

LEN = 4
_FIELD_MASK = 0x370000                      # LEN (bits 18:16) | STRIDE (bits 21:20)
_FIELD_SET = (LEN - 1) << 16                # STRIDE = 0b00 is stride 1

ENTER_MARK = "@ shortvec: enter"
LEAVE_MARK = "@ shortvec: leave"


# The markers bound the region lint checks: from after LEN is set to before it is cleared.
def enter(e, tmp="r12"):
    e(f"    vmrs {tmp}, fpscr")
    e(f"    bic {tmp}, {tmp}, #{_FIELD_MASK:#x}")
    e(f"    orr {tmp}, {tmp}, #{_FIELD_SET:#x}")
    e(f"    vmsr fpscr, {tmp}")
    e(f"    {ENTER_MARK} (LEN {LEN}, STRIDE 1)")


def leave(e, tmp="r12"):
    e(f"    {LEAVE_MARK}")
    e(f"    vmrs {tmp}, fpscr")
    e(f"    bic {tmp}, {tmp}, #{_FIELD_MASK:#x}")
    e(f"    vmsr fpscr, {tmp}")


# VFPv2 ops that LEN turns into vectors. Loads, stores, core<->VFP moves, vcmp and vcvt are always scalar.
_VECTOR_OPS = {"vadd", "vsub", "vmul", "vnmul", "vmla", "vmls", "vnmla", "vnmls", "vdiv", "vmov", "vabs", "vneg",
               "vsqrt"}
_EXITS = re.compile(r"^(bl|blx|bx|svc|swi)(\w\w)?(\.\w+)?$")
_SREG = re.compile(r"^s(\d+)$")


def _operands(rest):
    return [o.strip() for o in rest.split(",") if o.strip()]


def _check_vector(op, regs, where):
    """regs: operand strings of a .f32 data-processing op inside the region (dest first)."""
    s = [_SREG.match(r) for r in regs]
    if not all(s):                           # vmov sN, rM and the like are transfers, never vectors
        return
    n = [int(m.group(1)) for m in s]
    if n[0] < 8:                             # bank-0 destination: the whole op is scalar
        return
    # Sm (the last source) in bank 0 is a scalar operand; every other operand is a vector of LEN.
    vecs = n[:-1] if n[-1] < 8 else n
    for r in vecs:
        if r % 8 + LEN > 8:
            raise ValueError(f"{where}: {op} s{r} would wrap around its bank at LEN {LEN}")


def lint(lines):
    """Check one generated .S (a list of lines, or a string). Raises ValueError on the first violation."""
    if isinstance(lines, str):
        lines = lines.split("\n")
    lines = [x for chunk in lines for x in chunk.split("\n")]
    inside, func = False, None
    for i, raw in enumerate(lines, 1):
        line = raw.strip()
        where = f"line {i} ({func or '?'}): {line}"
        if line.startswith(ENTER_MARK):
            if inside:
                raise ValueError(f"{where}: enter inside a region")
            inside = True
            continue
        if line.startswith(LEAVE_MARK):
            if not inside:
                raise ValueError(f"{where}: leave outside a region")
            inside = False
            continue
        code = line.split("@")[0].strip()
        if not code:
            continue
        m = re.match(r"^\.size\s+(\w+)", code)
        if m:
            if inside:
                raise ValueError(f"{where}: function ends inside a region")
            func = None
            continue
        m = re.match(r"^(\w+):$", code)
        if m and not m.group(1).isdigit():
            func = m.group(1)
            continue
        if code.startswith(".") or code.endswith(":"):
            continue
        parts = code.split(None, 1)
        op, rest = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
        if not inside:
            continue
        if _EXITS.match(op):
            raise ValueError(f"{where}: leaves the function while LEN != 0")
        ops = _operands(rest)
        if (op.startswith(("pop", "ldm")) and re.search(r"\bpc\b", rest)) or (ops and ops[0] == "pc"):
            raise ValueError(f"{where}: writes pc while LEN != 0")
        if op == "vmsr":
            raise ValueError(f"{where}: FPSCR written inside a region")
        base, _, dt = op.partition(".")
        if base in _VECTOR_OPS:
            if dt == "f64":
                raise ValueError(f"{where}: double-precision op inside a region")
            if dt == "f32" or (not dt and rest.startswith("s")):
                _check_vector(op, _operands(rest), where)
    if inside:
        raise ValueError("end of file inside a region")


def _gate_kernel(e):
    # scripts/armv6_shortvec_gate.c's kernel: the GEMM tile's inner loop (gen_armv6_f32_tile.py) with the
    # loops of a driver around it, all inside one region, so a signal lands at any point of a call.
    #   sv_kernel(x [n][4], w [n][4], y [16] in/out, n, reps): y[c][t] += sum_k x[k][t] * w[k][c], reps times.
    #   sv_kernel_end marks the end of its code (the gate's handler tests the interrupted pc against both).
    e("    .global sv_kernel\n    .type sv_kernel, %function\nsv_kernel:")
    e("    push {r4-r6}")
    e("    vpush {d8-d15}")
    e("    ldr r4, [sp, #76]")      # reps (12 + 64 bytes pushed)
    enter(e)
    e("    vldmia r2, {s16-s31}")
    e("2:")
    e("    mov r5, r0\n    mov r6, r1\n    mov r12, r3")
    e("1:")
    e("    vldmia r5!, {s8-s11}")
    e("    vldmia r6!, {s0-s3}")
    e("    vmla.f32 s16, s8, s0")
    e("    vmla.f32 s20, s8, s1")
    e("    vmla.f32 s24, s8, s2")
    e("    vmla.f32 s28, s8, s3")
    e("    subs r12, r12, #1")
    e("    bne 1b")
    e("    subs r4, r4, #1")
    e("    bne 2b")
    e("    vstmia r2, {s16-s31}")
    leave(e)
    e("    vpop {d8-d15}")
    e("    pop {r4-r6}")
    e("    bx lr")
    e("    .global sv_kernel_end\nsv_kernel_end:")
    e("    .size sv_kernel, .-sv_kernel")


if __name__ == "__main__":
    import sys
    L = []
    e = L.append
    e("    .text\n    .arm\n    .fpu vfp\n    .align 4")
    _gate_kernel(e)
    e('    .section .note.GNU-stack,"",%progbits')
    lint(L)
    with open(sys.argv[1], "w") as f:
        f.write("\n".join(L) + "\n")
