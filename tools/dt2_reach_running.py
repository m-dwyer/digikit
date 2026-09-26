"""Bring a Digitakt II 1.16 ColdFire snapshot to a fully running state, in
bounded stages, saving a snapshot after each one so a later run never repeats
earlier work.

"Fully running" here means: the intro has handed over, the RTOS's own tasks
have spawned, and the display/timer-wheel/job-pump machinery is turning --
checked via task-control state (the switcher's ready list, `emu.tasks`) and
the same code-address marks `tools/bootcheck.py` uses to classify
MAIN_OS_RUNNING (mainloop/job_pump entered, PIT3/DTIM3 firing, vector 208
handed to the display module). No framebuffer/setPixel read is used for this
-- see emu/gui.py's own warning about that being the wrong instrument for the
main OS.

    uv run python tools/dt2_reach_running.py SNAPSHOT --out PREFIX \
        --stage N [--stage N ...] [--syx SYX] [--resume SNAP]

Each ``--stage N`` runs N further instructions (not cumulative) and saves
``PREFIX-stageK.snap`` (K = 1, 2, ...; the last stage's file is also copied to
``PREFIX.snap``). ``--resume SNAP`` continues from a snapshot this tool (or
``emu.checkpoint``) already produced, carrying its recorded instruction count
forward in the printed totals, instead of starting at ``boot400M.snap``.

The saved snapshots carry full checkpoint components (edma_tx, ssi0_dma if
present, esdhc, uart_in, timers) and the build manifest, via the same
``save_longrun``-style call ``emu.checkpoint`` uses -- so a later tool
(`tools/sharc_capture_run.py`, `emu.gui`) can resume one with
``emu.longrun.build`` exactly as it would ``boot400M.snap`` or a checkpoint
ladder rung.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unicorn import UC_HOOK_CODE  # noqa: E402

from emu import config, symbols  # noqa: E402
from emu.dtim import Dtims, Timers  # noqa: E402
from emu.longrun import build, spin  # noqa: E402
from emu.pit import Pits, intro_running  # noqa: E402
from emu.snapshot import save  # noqa: E402
from emu.tasks import ready_list  # noqa: E402

KIT_PTR = 0x80004704
KIT_LOAD_FN = 0x4002D9C4


def u32(m, a):
    try:
        return struct.unpack(">I", bytes(m.uc.mem_read(a, 4)))[0]
    except Exception:
        return None


def observe(m, ev, profile, mark, timers):
    """-> a dict of everything this tool checks, all read from task-control
    state / code-address marks / firmware globals -- never the framebuffer.

    ``timers`` is the combined ``emu.dtim.Timers`` object, whose ``.fired``
    property is string-keyed ("PIT3", "DTIM3", ...) -- see that class. Using
    the underlying ``Pits``/``Dtims`` objects' own int-keyed ``.fired``
    counters here directly was a bug: it always read 0, the same one
    `tools/bootcheck.py`'s own `run_arm` avoids by reading `pits.fired` off
    its `Timers` object under that same name.
    """
    cur = u32(m, profile.current_tcb) if profile.current_tcb else None
    rl = ready_list(m, profile.ready_cursor) if profile.ready_cursor else []
    vec208 = u32(m, 0x40000340)
    fired = timers.fired
    checks = {
        "mainloop entered": mark.get("mainloop", 0) > 0,
        "job pump running": mark.get("job_pump", 0) > 0,
        "PIT3 firing": fired.get("PIT3", 0) > 0,
        "DTIM3 firing": fired.get("DTIM3", 0) > 0,
        "vector 208 handed to display": vec208 != profile.intro_pit3_isr,
    }
    return {
        "tasks_created": len(ev["tasks"]),
        "distinct_tasks_scheduled": len(ev["switch"]),
        "current_tcb": cur,
        "ready_list_len": len(rl),
        "kit_ptr": u32(m, KIT_PTR),
        "kit_load_fn_hits": mark.get("kit_load_fn", 0),
        "checks": checks,
        "main_os_running": all(checks.values()),
    }


def run(snapshot, out_prefix, stages, syx=None, chunk=200_000, card_image=None):
    with open(config.main_image(), "rb") as fh:
        main_img = fh.read()
    profile = symbols.resolve(main_img)

    build_kwargs = dict(
        unblock=True,
        softfloat=True,
        bitmap=True,
        dsp=True,
        deferred_components=("timers",),
    )
    if syx:
        build_kwargs["syx"] = syx
    if card_image:
        build_kwargs["card_image"] = card_image
    m, ev, st, pc, inq, at = build(snapshot, **build_kwargs)

    # A snapshot this tool saved carries its own Pits/Dtims cadence (channels,
    # rate, held flag) as a deferred 'timers' component -- restore exactly
    # that instead of constructing fresh ones, or a resumed run would restart
    # the timer clock and could re-arm a channel the firmware had switched
    # off. Returns None for a snapshot with no such component (e.g. the
    # original boot ladder's boot400M.snap), in which case this is the FIRST
    # stage and a fresh intro-aware Timers is built the way emu/gui.py and
    # tools/bootcheck.py do.
    timers = ev["restore_checkpoint_timers"]()
    if timers is None:
        intro = intro_running(m, profile.intro_pit3_isr)
        pits = Pits(m, hold=intro)
        dtims = Dtims(m, channels=(3,), hold=intro)
        timers = Timers(pits, dtims)
        ev["checkpoint_components"]["timers"] = timers
    else:
        pits, dtims = timers.sources

    mark: dict[str, int] = {}

    def bump(key):
        def hook(uc, a, s_, d):
            mark[key] = mark.get(key, 0) + 1

        return hook

    if timers.held and profile.intro_done is not None:

        def handover(uc, a, s_, d):
            pits.release()
            dtims.release()
            mark["intro_done"] = mark.get("intro_done", 0) + 1

        at(profile.intro_done, handover)

    for name in ("mainloop", "job_pump", "task_start", "display_start"):
        addr = getattr(profile, name, None)
        if addr is not None:
            at(addr, bump(name))

    kit_hook = m.uc.hook_add(
        UC_HOOK_CODE, bump("kit_load_fn"), begin=KIT_LOAD_FN, end=KIT_LOAD_FN
    )

    base_n = st["n"]
    results = []
    last_path = None
    for i, n in enumerate(stages, 1):
        t0 = time.time()
        pc, done, stop = spin(m, pc, n, chunk, pits=timers)
        base_n += done
        st["n"] = base_n
        dt = time.time() - t0
        obs = observe(m, ev, profile, mark, timers)
        path = "%s-stage%d.snap" % (out_prefix, i)
        info = save(
            m,
            path,
            extra={"n": st["n"], "seen": sorted(st.get("seen", set()))},
            components={**ev["checkpoint_components"], "timers": timers},
            manifest=ev["checkpoint_manifest"],
        )
        last_path = path
        row = {
            "stage": i,
            "requested": n,
            "executed": done,
            "cum_instrs": st["n"],
            "stop": stop,
            "wall_s": round(dt, 1),
            "instrs_per_s": round(done / dt) if dt > 0 else None,
            "path": path,
            "bytes": info["bytes_on_disk"],
            **obs,
        }
        results.append(row)
        print(
            "[stage %d] +%d instrs (cum %d) in %.1fs (%.2fM/s) stop=%s "
            "tasks=%d ready=%d kit_ptr=%s kit_load_hits=%d main_os_running=%s"
            % (
                i,
                done,
                st["n"],
                dt,
                (done / dt / 1e6) if dt > 0 else 0.0,
                stop,
                row["tasks_created"],
                row["ready_list_len"],
                hex(row["kit_ptr"]) if row["kit_ptr"] else None,
                row["kit_load_fn_hits"],
                row["main_os_running"],
            ),
            flush=True,
        )
        if not row["main_os_running"]:
            print("    checks: %r  marks: %r" % (obs["checks"], dict(mark)), flush=True)
            print(
                "    pits: held=%s fired=%r missed=%r next=%r"
                % (pits.held, dict(pits.fired), dict(pits.missed), pits.next),
                flush=True,
            )
            print(
                "    dtims: held=%s fired=%r missed=%r next=%r"
                % (dtims.held, dict(dtims.fired), dict(dtims.missed), dtims.next),
                flush=True,
            )
        if pc == 0 or stop not in ("limit",):
            break

    m.uc.hook_del(kit_hook)
    if last_path is not None:
        final = out_prefix + ".snap"
        import shutil

        shutil.copyfile(last_path, final)
        print("final snapshot: %s" % final)

    m.close()
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot")
    p.add_argument("--out", required=True, help="output path prefix")
    p.add_argument(
        "--stage",
        dest="stages",
        action="append",
        type=lambda s: int(s, 0),
        required=True,
        help="instructions for one stage (repeatable, in order)",
    )
    p.add_argument("--syx")
    p.add_argument("--chunk", type=lambda s: int(s, 0), default=200_000)
    p.add_argument(
        "--card-image",
        dest="card_image",
        default=None,
        help="+Drive image built by tools/plusdrive.py to serve behind the "
        "eSDHC/eMMC model (see emu.esdhc.Card.from_file); default: the "
        "blank, all-zero card",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(
        args.snapshot,
        args.out,
        args.stages,
        syx=args.syx,
        chunk=args.chunk,
        card_image=args.card_image,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
