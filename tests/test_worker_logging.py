"""Regression coverage for Python/native output in reused spawned workers."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _raw_stream_type(stream):
    buffer = getattr(stream, "buffer", stream)
    return type(getattr(buffer, "raw", buffer)).__name__


def _worker_probe(directory, fail, native_solver):
    from deterministic_optimization.cli import run_one

    original = sys.stdout, sys.stderr
    stream_type = _raw_stream_type(sys.stdout)
    directory = Path(directory)
    directory.mkdir()

    def fake_run(output, **kwargs):
        print("Python stdout: transfer \u2192 Psyche", flush=True)
        print("Python stderr", file=sys.stderr, flush=True)
        os.write(1, b"native stdout\n")
        os.write(2, b"native stderr\n")
        if native_solver:
            # Exercise the actual CasADi/IPOPT output path without a mission solve.
            import casadi as ca

            problem = ca.Opti()
            x = problem.variable()
            problem.minimize((x - 2) ** 2)
            problem.solver("ipopt", {"print_time": False}, {"print_level": 5, "sb": "yes"})
            solution = problem.solve()
            assert abs(float(solution.value(x)) - 2) < 1e-8
        if fail:
            raise RuntimeError("intentional worker failure")
        # This logging probe must not publish mission outputs.
        return {"converged": False}

    with patch("deterministic_optimization.psyche_driver.run", side_effect=fake_run):
        result = run_one("psyche", "fuel", directory, 0, directory / "public")
    assert sys.stdout is original[0] and sys.stderr is original[1]
    print("stdout restored after worker", flush=True)
    print("stderr restored after worker", file=sys.stderr, flush=True)
    return {"pid": os.getpid(), "stream_type": stream_type, "result": result}


def _check_spawned_workers(directory):
    records = []
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for name, fail, native in (
            ("first", False, True),
            ("failure", True, False),
            ("reused", False, False),
        ):
            folder = Path(directory) / name
            record = pool.submit(_worker_probe, str(folder), fail, native).result(timeout=90)
            contents = (folder / "psyche_fuel.log").read_text(encoding="utf-8")
            for marker in (
                "transfer \u2192 Psyche",
                "Python stderr",
                "native stdout",
                "native stderr",
            ):
                assert marker in contents, (name, marker, contents)
            if native:
                assert "EXIT: Optimal Solution Found." in contents, contents
            if fail:
                assert "intentional worker failure" in record["result"]["error"]
                assert "RuntimeError: intentional worker failure" in contents
            else:
                assert "error" not in record["result"], record
            assert "WinError" not in contents, contents
            assert not (folder / "public").exists()
            records.append(record)
    assert len({record["pid"] for record in records}) == 1
    return records


class WorkerLoggingChecks(unittest.TestCase):
    def test_spawned_worker_captures_python_native_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_spawned_workers(directory)

    @unittest.skipUnless(os.name == "nt", "Windows console regression")
    def test_real_windows_console_streams_in_spawned_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            # No redirected handles: exercise WindowsConsoleIO, not pipe FileIO.
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--console-probe", directory],
                cwd=ROOT,
                startupinfo=startup,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                timeout=120,
                check=False,
            )
            report = json.loads((Path(directory) / "console_report.json").read_text())
            self.assertEqual(process.returncode, 0, report)
            self.assertIn("WindowsConsoleIO", report["parent_stream_type"])
            for record in report["records"]:
                self.assertIn("WindowsConsoleIO", record["stream_type"])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--console-probe":
        directory = Path(sys.argv[2])
        report = {"parent_stream_type": _raw_stream_type(sys.stdout)}
        status = 0
        try:
            report["records"] = _check_spawned_workers(directory)
        except Exception:
            report["error"] = traceback.format_exc()
            status = 1
        (directory / "console_report.json").write_text(json.dumps(report, indent=2))
        raise SystemExit(status)
    unittest.main()
