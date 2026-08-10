================================================================================
 HOW TO RUN AN ARCH TEST MANUALLY (no run_arch_test.py)
================================================================================

This explains, step by step, what run_arch_test.py automates -- so if you're
asked to run a test "by hand" and show the output, you know exactly which
commands to type and what they mean.

Everything below uses M-div-00.S as the example. To run a different test,
just swap TEST/NAME for the file you want (see "Which test files exist?" at
the bottom).

All commands assume your current directory is:
  32I/tests/riscv-arch-test/


--------------------------------------------------------------------------------
 THE BIG PICTURE (read this first)
--------------------------------------------------------------------------------

Each .S file (e.g. M-div-00.S) is one official riscv-arch-test test: hundreds
of tiny test cases back to back in one assembly file, each one doing a single
instruction (e.g. `div x18, x0, x21`) and then recording the result into a
block of memory called the "signature region" (bounded by two labels,
begin_signature and end_signature).

The idea: run the SAME test body TWICE --
  1. once on OUR OWN emulator (this repo's CPU.cpp/memory.cpp)
  2. once on a real, independently-written RISC-V implementation (QEMU)
...then compare the two signature dumps word-for-word. If every word matches,
our emulator computed the exact same answer as a real RISC-V chip would, for
every one of those hundreds of test cases.

We don't use the official riscv-arch-test framework's own machinery for this
(it needs the Sail reference model + a Ruby/UDB toolchain + hours of setup --
see the chat history for why that was ruled out). Instead:

  - env/riscv_arch_test.h is a small HAND-WRITTEN replacement for the
    framework's real header. It defines just enough (RVTEST_BEGIN,
    RVTEST_SIGUPD, LA, LI, SREG, LREG, etc.) for these test files to compile
    and run standalone, without CSRs, traps, or privilege modes -- this
    emulator doesn't implement any of that, and these tests don't need it.

  - link_arch_test.ld places everything at address 0x10000 (not 0x0).
    This is NOT something the test requires -- it's purely because QEMU runs
    the binary as a real Linux process, and Linux refuses to map a program at
    address 0 (the reserved null page). Our own emulator has no such
    restriction, but both builds have to use the SAME address, otherwise
    anything that computes an absolute address (auipc, jal's return address)
    would disagree between the two runs for a reason that has nothing to do
    with whether the CPU is actually correct.

  - The test body itself doesn't know or care which "backend" it's running
    on. The only difference between the two builds is what happens at the
    very end (RVTEST_CODE_END, in env/riscv_arch_test.h):
      normal build:          spin in an infinite self-loop (`j self`).
                              Our emulator detects this (CPU::is_halted())
                              and stops running.
      -DQEMU_REFERENCE build: write the whole signature region to stdout
                              (a Linux `write` syscall) and exit(0), so
                              QEMU produces a clean, capturable byte stream.


--------------------------------------------------------------------------------
 STEP 1: Build the harness (only needed once, or after editing CPU.cpp)
--------------------------------------------------------------------------------

This compiles harness.cpp together with this emulator's real CPU.cpp and
memory.cpp -- it's not a separate reimplementation, it's the actual emulator.

  g++ -Wall -Wextra -std=c++11 -O2 -o build/arch_test_harness \
    harness.cpp ../basic/loader.cpp ../../src/CPU.cpp ../../src/memory.cpp

No output means success. You'll get build/arch_test_harness.


--------------------------------------------------------------------------------
 STEP 2: Cross-compile the test for OUR emulator
--------------------------------------------------------------------------------

  TEST=assembly_files/rv32i/M/M-div-00.S
  NAME=M-div-00

  riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -nostartfiles \
    -ffreestanding -O0 -I env -T link_arch_test.ld -o build/$NAME.elf $TEST

You may see a harmless linker warning ("has a LOAD segment with RWX
permissions") -- ignore it, that's normal for a freestanding bare-metal ELF
with no separate code/data protection.


--------------------------------------------------------------------------------
 STEP 3: Find the signature region's address
--------------------------------------------------------------------------------

Every test file defines begin_signature/end_signature labels bounding the
memory it writes results into. `nm` lists ELF symbols with their addresses:

  riscv64-unknown-elf-nm build/$NAME.elf | grep -E "begin_signature|end_signature"

Example output:
  00019ac8 D begin_signature
  0001a110 D end_signature

Write these down (without the leading zeros is fine) -- you need them for
step 5. Note they're specific to each test file (different tests have
different amounts of code before the signature region, so the address
shifts).


--------------------------------------------------------------------------------
 STEP 4: Strip the ELF to a raw binary
--------------------------------------------------------------------------------

Our emulator's loader (loader.cpp) just reads a flat byte blob into memory --
it doesn't understand ELF headers/sections, so objcopy has to strip those out
first:

  riscv64-unknown-elf-objcopy -O binary build/$NAME.elf build/$NAME.bin


--------------------------------------------------------------------------------
 STEP 5: Run it on OUR emulator
--------------------------------------------------------------------------------

  ./build/arch_test_harness build/$NAME.bin <sig_start_hex> <sig_end_hex> [max_cycles]

Using the addresses from step 3 (without "0x", the harness parses them as
hex):

  ./build/arch_test_harness build/M-div-00.bin 19ac8 1a110 200000

What happens: it loads the binary at address 0x10000 (matching where the
linker placed it), runs the CPU cycle by cycle until it hits the self-loop
halt (or 200000 cycles, whichever comes first -- 200000 is just a generous
safety cap, real tests finish in a few hundred to a few thousand cycles), then
prints every 32-bit word from sig_start to sig_end, one per line, in hex.

Example output (first few lines):
  [Loader] Loaded 41232 bytes from build/M-div-00.bin into memory at 0x10000
  00000000
  00000001
  00000000
  00000000
  ...
(402 hex-word lines total for M-div-00.S, after that one [Loader] line.)


--------------------------------------------------------------------------------
 STEP 6: Cross-compile the SAME test for the QEMU reference
--------------------------------------------------------------------------------

Identical command to step 2, plus -DQEMU_REFERENCE (this is what switches
RVTEST_CODE_END from "loop forever" to "write results + exit", see env/
riscv_arch_test.h):

  riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -nostartfiles \
    -ffreestanding -O0 -DQEMU_REFERENCE -I env -T link_arch_test.ld \
    -o build/${NAME}_qemu_ref.elf $TEST


--------------------------------------------------------------------------------
 STEP 7: Run it under QEMU
--------------------------------------------------------------------------------

  qemu-riscv32 build/${NAME}_qemu_ref.elf > /tmp/qemu_out.bin

qemu-riscv32 runs the ELF as if it were a real Linux process (user-mode
emulation of a real RISC-V CPU -- this is a genuine, independent RISC-V
implementation, not related to our CPU.cpp at all). Because of the
-DQEMU_REFERENCE write-syscall trick, its stdout IS the raw signature bytes
(not text) -- 402 words x 4 bytes = 1608 bytes for M-div-00.S.

To view it as hex words (one per line, matching step 5's format):

  od -An -tx4 -v --endian=little /tmp/qemu_out.bin | tr -s ' ' '\n' | grep -v '^$'

Example output (first few lines -- notice these match step 5 exactly):
  00000000
  00000001
  00000000
  00000000
  ...


--------------------------------------------------------------------------------
 STEP 8: Compare
--------------------------------------------------------------------------------

If you saved both dumps to files, `diff` them directly:

  ./build/arch_test_harness build/$NAME.bin 19ac8 1a110 200000 \
    | grep -E '^[0-9a-f]{8}$' > /tmp/ours.txt
  od -An -tx4 -v --endian=little /tmp/qemu_out.bin | tr -s ' ' '\n' \
    | grep -v '^$' > /tmp/qemu.txt
  diff /tmp/ours.txt /tmp/qemu.txt && echo "IDENTICAL -- test passes"

No output from `diff` (and "IDENTICAL" printed) means every one of the ~400
test cases in this file produced the exact same result on our emulator as on
real QEMU. Any diff output means at least one test case disagrees -- that
line number (minus the [Loader] offset) tells you which test case, and you
can look it up by counting RVTEST_SIGUPD calls in the .S file, or just rerun
run_arch_test.py, which prints "test N: expected 0x..., got 0x..." directly.


--------------------------------------------------------------------------------
 What run_arch_test.py actually does
--------------------------------------------------------------------------------

It's exactly steps 1-8 above, automated, for EVERY .S file found under
assembly_files/ (not just one):

  1. build_cpp_harness()       -- step 1
  2. discover_tests()          -- finds every .S file under assembly_files/
     then for each one:
  3. build_riscv_elf()         -- step 2
  4. signature_bounds()        -- step 3 (parses `nm` output itself)
  5. to_bin()                  -- step 4
  6. run_harness()             -- step 5 (parses out just the hex-word lines)
  7. build_qemu_reference_elf()-- step 6
  8. run_qemu_reference()      -- step 7 (reads raw stdout bytes directly,
                                   no od/tr needed -- Python's struct.unpack
                                   does the same job)
  9. word-by-word comparison   -- step 8, printing "test N: expected/got" for
                                   any mismatch instead of a raw diff

Run it with:

  python3 run_arch_test.py

It builds everything, runs every test, and prints PASS/FAIL per file plus a
final "ALL TESTS PASSED" / "SOME TESTS FAILED" line, with exit code 0 or 1
(useful for CI/scripting -- `python3 run_arch_test.py && echo ok`).


--------------------------------------------------------------------------------
 Which test files exist?
--------------------------------------------------------------------------------

  find assembly_files -name "*.S"

Currently: the base RV32I integer instructions (assembly_files/rv32i/I/) and
the M-extension multiply/divide instructions (assembly_files/rv32i/M/).
Byte/halfword/word LOAD tests (I-lb/lbu/lh/lhu/lw-00.S) are NOT included --
they reference a "scratch" memory region our env header doesn't define yet.

To add more test suites: copy a folder from riscv-arch-test/tests/<suite>/
into assembly_files/<suite>/ (same relative path) and run
run_arch_test.py again -- discover_tests() picks up new files automatically,
no code changes needed, AS LONG AS the test only uses macros env/
riscv_arch_test.h already defines. If a new test fails to even compile
("unrecognized opcode" or "undefined reference"), grep the .S file for
RVTEST_/uppercase macro calls and check whether env/riscv_arch_test.h defines
them -- if not, that's a new macro to add (see how LA/LI/SREG/LREG/REGWIDTH
were added, following tests/env/utils.h in the sibling riscv-arch-test repo
for the official definition to copy from).
