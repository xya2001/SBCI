"""How long each registration of the eleven takes, onto their template and onto the first.

    SBCI_HCP_DIR=hcp-ya python tests/reference/registration_timing_probe.py DEVICE METHOD OUT.json

DEVICE is cpu or gpu; METHOD is encore, conseal or both. The eleven released young adults
(``SBCI_HCP_DIR``) are aligned as the cohort figure aligns them -- ENCORE on their densities
re-smoothed through the package's kernel, ten iterations and four backtracks; ConSEAL on
their endpoints, thirty iterations at its defaults -- twice: onto the template estimated
from all eleven (``template=None``), and onto the first subject (``template=`` that
subject). Every call of the engines' ``template`` and ``register`` and of the ConSEAL
kernel's construction is timed, the GPU synchronized before and after, with the steps each
registration took, into OUT.json; a summary is printed. Run with ``PYTHONPATH`` at the
tree's ``src`` (PORTING.md item 28, *A registration's time*).
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

import sbci
from sbci import _device
from sbci.alignment import Encore
from sbci.conseal import ConSEAL, EndpointConnectome, HeatKernelBuilder, default_grids

sys.path.insert(0, str(Path(sbci.__file__).resolve().parents[2] / "scripts"))
from make_figures import resmoothed  # noqa: E402

CALLS = []


def timed(cls, name, label):
    original = getattr(cls, name)

    def wrapper(self, *args, **kwargs):
        device = getattr(self, "device", "cpu")
        _device.synchronize(device)
        start = time.perf_counter()
        out = original(self, *args, **kwargs)
        _device.synchronize(device)
        CALLS.append((label, time.perf_counter() - start))
        return out

    setattr(cls, name, wrapper)


timed(Encore, "template", "template")
timed(Encore, "register", "register")
timed(ConSEAL, "template", "template")
timed(ConSEAL, "register", "register")
timed(HeatKernelBuilder, "compute", "kernel")


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def summary(steps):
    """The calls made since the last summary, with each registration's steps."""
    calls = list(CALLS)
    CALLS.clear()
    registers = [t for label, t in calls if label == "register"]
    return {
        "template_s": sum(t for label, t in calls if label == "template"),
        "kernel_s": sum(t for label, t in calls if label == "kernel"),
        "register_s": registers,
        "steps": [int(s) for s in steps],
    }


def encore(device, subjects, out):
    begun = time.perf_counter()
    given = [resmoothed(s, s.endpoints) for s in subjects]
    out["encore_resmooth_s"] = time.perf_counter() - begun
    for kind, template in (("template", None), ("first", given[0])):
        begun = time.perf_counter()
        result = sbci.align(
            given, template=template, max_iterations=10, backtracks=4, device=device
        )
        total = time.perf_counter() - begun
        out[f"encore_{kind}"] = summary([len(trace) - 1 for trace in result.traces])
        out[f"encore_{kind}"]["total_s"] = total
        print(f"{device} ENCORE onto the {kind}: {total:.1f} s", flush=True)
        del result


def conseal(device, subjects, out):
    lh, rh = default_grids()
    carriers = [EndpointConnectome.from_endpoints(s.endpoints, lh, rh) for s in subjects]
    for kind, template in (("template", None), ("first", carriers[0])):
        begun = time.perf_counter()
        result = sbci.endpoints_align(carriers, template=template, max_iterations=30, device=device)
        total = time.perf_counter() - begun
        out[f"conseal_{kind}"] = summary([len(costs) - 1 for costs in result.costs])
        out[f"conseal_{kind}"]["total_s"] = total
        print(f"{device} ConSEAL onto the {kind}: {total:.1f} s", flush=True)
        del result


def main(device, method, path):
    paths = sorted(Path(os.environ["SBCI_HCP_DIR"]).glob("sub-*_sc.h5"))
    subjects = [sbci.load(p) for p in paths]
    out = {
        "device": device,
        "host": platform.node(),
        "cpu": cpu_name(),
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "subjects": [p.name.split("_")[0] for p in paths],
    }
    if device == "gpu":
        out["gpu"] = _device.cupy().cuda.runtime.getDeviceProperties(0)["name"].decode()
    if method in ("encore", "both"):
        encore(device, subjects, out)
    if method in ("conseal", "both"):
        conseal(device, subjects, out)
    Path(path).write_text(json.dumps(out, indent=1))
    for key, value in out.items():
        if isinstance(value, dict):
            times = np.asarray(value["register_s"])
            print(
                f"  {key}: template {value['template_s']:.1f} s, kernel {value['kernel_s']:.1f} s, "
                f"{len(times)} registrations, mean {times.mean():.2f} s, median "
                f"{np.median(times):.2f} s, {times.min():.2f} to {times.max():.2f} s; steps "
                f"{value['steps']}; in all {value['total_s']:.1f} s"
            )


if __name__ == "__main__":
    main(*sys.argv[1:4])
