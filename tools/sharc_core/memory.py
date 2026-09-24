"""Data memory reads and writes, DAG address arithmetic and SIMD companions.

Moved verbatim from tools/sharc_trace.py.
"""

from __future__ import annotations

from typing import Optional

from sharcimm import name_address
from sharcldr import SW_ALIAS_BASE, sw_to_byte

from .encoding import (
    CORE_MMR_RESET_VALUES,
    L1_BLOCK3_NW_BASE,
    L1_BLOCK3_NW_LIMIT,
    L1_BLOCK3_SW_BASE,
    UREG_CODES,
    UREG_NAMES,
)
from .values import (
    Const,
    Unknown,
    Value,
    _add,
)
from .state import (
    State,
    _cureg_code,
    _json_value,
    _render,
    _simd_active,
)


def _concrete_address(value: Value | int) -> Optional[int]:
    return (
        value.value
        if isinstance(value, Const)
        else (value if isinstance(value, int) else None)
    )


def _canonical_dm_address(
    state: State, address: int, width: int, *, for_write: bool = False
) -> Optional[int]:
    """Resolve a DSP DM address to the loader's byte-address alias.

    Application code uses unaliased DM pointers such as ``0x26968c`` whereas
    the boot stream is keyed at ``SW_ALIAS_BASE + 0x26968c``.  Keep an already
    mapped direct address (notably external memory and MMRs) unchanged; only
    retry an unmapped low address through the alias.
    """
    concrete = state.concrete
    if concrete is None:
        return None

    def mapped(base: int) -> bool:
        return all(
            here in state.overlay or concrete.read(here, 1) is not None
            for here in range(base, base + width)
        )

    if mapped(address):
        return address
    if 0 <= address < SW_ALIAS_BASE:
        alias = SW_ALIAS_BASE + address
        if for_write or mapped(alias):
            return alias
    # Runtime RAM and MMR destinations need not have loader initializer bytes.
    # A concrete write creates those bytes in this path's overlay.
    return address if for_write else None


def _dm_read(
    state: State, address: Value | int, width: int, signed: bool = False
) -> Optional[Const]:
    """Read little-endian loader-backed DM bytes plus this path's overlay."""
    concrete = _concrete_address(address)
    if state.concrete is None or concrete is None or width not in (1, 2, 4, 8):
        return None
    fixed_width_mmr = (
        concrete in CORE_MMR_RESET_VALUES or name_address(concrete) is not None
    )
    if width == 4 and fixed_width_mmr and concrete in state.mmrs:
        value = state.mmrs[concrete]
        return value if isinstance(value, Const) else None
    if width == 4 and fixed_width_mmr and state.data_memory_tainted:
        return None
    if (
        width == 4
        and not state.assume_nw32
        and not fixed_width_mmr
        and not 0x30000000 <= concrete < 0x40000000
    ):
        # Internal normal-word width depends on runtime IMDWx state.  Reading
        # four loader bytes as one word is opt-in until that state is known.
        return None
    concrete = _canonical_dm_address(state, concrete, width)
    if concrete is None:
        return None
    if state.data_memory_tainted and not all(
        here in state.overlay for here in range(concrete, concrete + width)
    ):
        return None
    backing = state.concrete
    assert backing is not None
    raw = bytearray()
    for here in range(concrete, concrete + width):
        if here in state.overlay:
            raw.append(state.overlay[here])
        else:
            byte = backing.read(here, 1)
            assert byte is not None
            raw.append(byte[0])
    value = int.from_bytes(raw, "little", signed=signed)
    # A long word needs a register pair, which this tracer intentionally does
    # not model.  Do not truncate it into a false 32-bit value.
    return Const(value) if width <= 4 else None


def _read_px48(state: State, address: Value | int) -> Optional[tuple[Const, Const]]:
    """Read a loader-backed 48-bit normal word into the PX1/PX2 halves.

    A combined-PX DM or PM transfer without ``LW`` is 48 bits.  L1 block 3's
    normal-word alias packs those words in three 16-bit columns, while loader
    records use the short-word/system-byte view.  Each 48-bit word therefore
    consumes six loader bytes.  The three parcels are individually little-
    endian, but retain their architectural high-to-low order.
    """
    concrete = _concrete_address(address)
    if (
        state.concrete is None
        or concrete is None
        or not L1_BLOCK3_NW_BASE <= concrete < L1_BLOCK3_NW_LIMIT
    ):
        return None
    offset = concrete - L1_BLOCK3_NW_BASE
    byte_address = sw_to_byte(L1_BLOCK3_SW_BASE) + 6 * offset
    raw = state.concrete.read(byte_address, 6)
    if raw is None:
        return None
    high, middle, low = (
        int.from_bytes(raw[start : start + 2], "little") for start in (0, 2, 4)
    )
    px2 = Const((high << 16) | middle)
    px1 = Const(low << 16)
    return px1, px2


def _load_normal_ureg(
    state: State, space: str, address: Value | int, code: int
) -> Optional[Const | dict[str, int]]:
    """Load one normal-word UREG value, including combined-PX DM/PM reads."""
    if code == UREG_CODES["PX"]:
        halves = _read_px48(state, address)
        if halves is not None:
            px1, px2 = halves
            state.uregs[UREG_CODES["PX"]] = Unknown(
                "combined PX represented by PX1/PX2"
            )
            state.uregs[UREG_CODES["PX1"]] = px1
            state.uregs[UREG_CODES["PX2"]] = px2
            return {"PX1": px1.value, "PX2": px2.value}
        state.uregs[UREG_CODES["PX1"]] = Unknown("memory-address " + _render(address))
        state.uregs[UREG_CODES["PX2"]] = Unknown("memory-address " + _render(address))
    elif space == "DM":
        loaded = _dm_read(state, address, 4)
        state.uregs[code] = loaded or Unknown("memory-address " + _render(address))
        return loaded
    state.uregs[code] = Unknown("memory-address " + _render(address))
    return None


def _dm_write(state: State, address: Value | int, width: int, value: Value) -> bool:
    concrete = _concrete_address(address)
    if (
        state.concrete is None
        or concrete is None
        or not isinstance(value, Const)
        or width not in (1, 2, 4)
    ):
        return False
    fixed_width_mmr = (
        concrete in CORE_MMR_RESET_VALUES or name_address(concrete) is not None
    )
    if width == 4 and fixed_width_mmr:
        state.mmrs[concrete] = value
        return True
    if (
        width == 4
        and not state.assume_nw32
        and not fixed_width_mmr
        and not 0x30000000 <= concrete < 0x40000000
    ):
        return False
    concrete = _canonical_dm_address(state, concrete, width, for_write=True)
    if concrete is None:
        return False
    raw = (value.value & 0xFFFFFFFF).to_bytes(4, "little")[:width]
    state.overlay.update(zip(range(concrete, concrete + width), raw))
    return True


def _dossier(state: State, target: int, return_sw: int) -> dict:
    registers = {
        UREG_NAMES[k]: _json_value(v)
        for k, v in state.uregs.items()
        if isinstance(v, Const)
    }
    objects = []
    if state.concrete is not None and state.dossier_bytes:
        seen = set()
        for name, value in registers.items():
            if not isinstance(value, int) or value in seen:
                continue
            raw = bytearray()
            for offset in range(state.dossier_bytes):
                b = _dm_read(state, value + offset, 1)
                if b is None:
                    break
                raw.append(b.value)
            if raw:
                seen.add(value)
                objects.append(
                    {
                        "register": name,
                        "address": value,
                        "bytes": list(raw),
                        "words_le": [
                            int.from_bytes(raw[i : i + 4], "little")
                            for i in range(0, len(raw) - 3, 4)
                        ],
                    }
                )
    return {
        "target_sw": target,
        "return_sw": return_sw,
        "registers": registers,
        "objects": objects,
    }


# Widths this tracer resolves a SIMD companion memory transfer for (SHARC+
# PRM p.212, Table 6-10 "DAG Address vs. Access Modes": explicit address
# Ia, implicit address Ia+k, k=1 for normal-word). Byte and short-word
# access have their own SIMD addressing rule (PRM pp.222-223, packing the
# companion into an adjacent byte/short-word rather than offsetting by a
# whole normal word) that this tracer does not yet model, so those widths
# raise rather than silently transfer only the explicit half.
_SIMD_COMPANION_WIDTHS = frozenset({"normal-word"})


def _simd_ureg_mem_companion(
    state: State, code: int, address: Value, access_width: str = "normal-word"
) -> Optional[tuple[int, Value]]:
    """The SIMD companion (Cureg code, companion address) for a single
    UREG<->memory transfer, or None when no companion transfer applies
    (SISD mode, or a UREG with no SIMD complement -- SHARC+ PRM p.15-12's
    TCOUNT/USTAT1 worked example: only "Cureg subset registers" gain a
    companion in SIMD mode, every other UREG "operates the same in SIMD
    and SISD mode").

    An unresolved MODE1.PEYEN is treated like SISD (no companion): the
    explicit transfer this helper's caller performs regardless is correct
    either way (SIMD only ever *adds* an implicit transfer beside it, PRM
    p.28), so an unknown MODE1 cannot make the explicit half wrong -- it
    can only leave a companion transfer unmodelled, exactly as this
    tracer already left it before this helper existed.  ACCESS_WIDTH not
    in _SIMD_COMPANION_WIDTHS raises ValueError instead, since a
    concretely SIMD-active companion this tracer cannot model would
    otherwise be silently dropped; callers should stop the state on that.
    """
    cureg = _cureg_code(code)
    if cureg is None:
        return None
    if access_width == "long-word":
        # PRM p.13-17: the (LW) modifier "override[s] SIMD mode, so these
        # loads always operate in SISD mode" -- unlike byte/short-word,
        # this is a documented no-companion case, not an unmodelled one.
        return None
    if _simd_active(state) is not True:
        return None
    if access_width not in _SIMD_COMPANION_WIDTHS:
        raise ValueError(
            "unsupported SIMD companion access width %r for %s"
            % (access_width, UREG_NAMES[code])
        )
    k = _access_modifier_scale("normal-word", state.assume_nw32)
    companion_address = _add(address, Const(k), "%s + %d" % (_render(address), k))
    return cureg, companion_address


def _access_modifier_scale(access_width: str, assume_nw32: bool) -> int:
    """Return SHARC+ byte-space scaled-address arithmetic width."""
    if access_width.startswith("short-word"):
        return 2
    if access_width == "long-word":
        return 8
    if access_width == "normal-word" and assume_nw32:
        return 4
    return 1


def _circular_wrap_const(index: int, base: int, length: int, delta: int) -> Optional[int]:
    """Concrete DAG circular-buffer wrap: the true I in [B, B+L) advances by
    DELTA and is corrected by one +-length step when it leaves the buffer
    (SHARC+ Core Programming Reference, out/refs/sharc-plus-prm, Sec. 6
    "Circular Buffering": "If the index pointer falls outside the buffer,
    the DAG subtracts or adds the buffer length to the index value,
    wrapping the index pointer back within the start and end boundaries of
    the buffer"). Valid for a modifier magnitude up to and including LENGTH
    (an exact-L step lands back on INDEX, which is correct: a full lap of a
    circular buffer is the identity). Returns None when |delta| exceeds
    LENGTH, since a single +-length correction is no longer guaranteed to
    land back in range and this module does not model multi-lap modifies.
    """
    if length <= 0 or abs(delta) > length:
        return None
    candidate = (index + delta) & 0xFFFFFFFF
    lower, upper = base, base + length
    if candidate < lower:
        candidate += length
    elif candidate >= upper:
        candidate -= length
    return candidate
