"""SHARC+ core semantics shared by the symbolic tracer (tools/sharc_trace.py)
and the concrete runner (tools/sharc_run.py).

Layers, lowest first; a module imports only from modules before it:
  encoding   Register codes, reset values, flag bits, access widths and instruction-field access.
  values     Concrete and symbolic values and the integer arithmetic over them.
  state      Machine state, register access and the trace event log.
  memory     Data memory reads and writes, DAG address arithmetic and SIMD companions.
  floats     32-bit IEEE float and fixed/float conversion helpers.
  flags      ASTATX flag updates for compute results.
  compute    Compute operations (ALU, multiplier, shifter, multifunction) and their application.
  sequencer  Decode, program flow, conditions, delayed branches, calls, returns and loops.
  forms      Per-form instruction execution: the _execute dispatch.
"""
