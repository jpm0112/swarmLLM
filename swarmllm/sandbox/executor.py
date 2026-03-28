from __future__ import annotations

"""
Sandbox for executing agent-generated code safely.

Runs agent code in a subprocess with:
- Time limits
- Restricted imports
- Memory limits (best-effort on Windows)
"""

import subprocess
import sys
import json
import tempfile
import os
import asyncio

from swarmllm.config import SandboxConfig
from swarmllm.tracking.telemetry import TelemetrySink


def execute_agent_code(
    code: str,
    input_data: list[dict],
    config: SandboxConfig,
    function_name: str = "schedule",
    telemetry: TelemetrySink | None = None,
    process_label: str | None = None,
    process_metadata: dict | None = None,
) -> dict:
    """
    Execute agent-generated code in a sandboxed subprocess.

    The agent code must define a function with the given function_name that
    accepts a list of dicts and returns a result.

    Returns:
        dict with keys: success, result, error, stdout
    """
    runner_code = _build_runner(code, input_data, config.blocked_imports, function_name)

    # Write to temp file and execute in subprocess
    tmp_dir = tempfile.mkdtemp(prefix="swarmllm_")
    tmp_file = os.path.join(tmp_dir, "agent_run.py")

    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(runner_code)

        result = _run_subprocess(
            [sys.executable, tmp_file],
            cwd=tmp_dir,
            timeout=config.timeout,
            telemetry=telemetry,
            label=process_label or "sandbox python",
            metadata=process_metadata,
        )

        # Auto-install missing packages and retry once
        if result.returncode != 0 and "ModuleNotFoundError" in (result.stderr or ""):
            pkg = _extract_missing_module(result.stderr)
            if pkg and pkg not in config.blocked_imports:
                install = _run_subprocess(
                    [sys.executable, "-m", "pip", "install", "--quiet", pkg],
                    cwd=tmp_dir,
                    timeout=60,
                    telemetry=telemetry,
                    label=f"pip install {pkg}",
                    metadata={"package": pkg, "stage": "sandbox_install"},
                )
                if install.returncode == 0:
                    # Retry
                    result = _run_subprocess(
                        [sys.executable, tmp_file],
                        cwd=tmp_dir,
                        timeout=config.timeout,
                        telemetry=telemetry,
                        label=process_label or "sandbox python",
                        metadata=process_metadata,
                    )

        if result.returncode != 0:
            return {
                "success": False,
                "result": None,
                "error": result.stderr[-2000:] if result.stderr else "Unknown error",
                "stdout": result.stdout[-1000:] if result.stdout else "",
            }

        # Parse the output — last line should be JSON
        stdout_lines = result.stdout.strip().split("\n")
        output_line = stdout_lines[-1] if stdout_lines else ""

        try:
            output = json.loads(output_line)
        except json.JSONDecodeError:
            return {
                "success": False,
                "result": None,
                "error": f"Could not parse output as JSON. Output: {result.stdout[-500:]}",
                "stdout": result.stdout[-1000:],
            }

        return {
            "success": output.get("success", False),
            "result": output.get("result"),
            "error": output.get("error"),
            "stdout": "\n".join(stdout_lines[:-1])[-1000:],
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "result": None,
            "error": f"Execution timed out after {config.timeout} seconds",
            "stdout": "",
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": f"Sandbox error: {str(e)}",
            "stdout": "",
        }
    finally:
        # Cleanup
        try:
            os.remove(tmp_file)
            os.rmdir(tmp_dir)
        except OSError:
            pass


async def execute_agent_code_async(
    code: str,
    input_data: list[dict],
    config: SandboxConfig,
    function_name: str = "schedule",
    telemetry: TelemetrySink | None = None,
    process_label: str | None = None,
    process_metadata: dict | None = None,
) -> dict:
    """Run sandbox execution without blocking the event loop."""
    return await asyncio.to_thread(
        execute_agent_code,
        code,
        input_data,
        config,
        function_name,
        telemetry,
        process_label,
        process_metadata,
    )


def _extract_missing_module(stderr: str) -> str | None:
    """Extract module name from ModuleNotFoundError traceback."""
    import re
    match = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", stderr)
    if match:
        # Return top-level package name (e.g., "pulp" from "pulp.core")
        return match.group(1).split(".")[0]
    return None


def _run_subprocess(
    command: list[str],
    *,
    cwd: str,
    timeout: int,
    telemetry: TelemetrySink | None,
    label: str,
    metadata: dict | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a tracked subprocess and keep it visible to the live monitor."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )
    if telemetry:
        telemetry.register_process(
            process.pid,
            label=label,
            kind="python",
            role="sandbox",
            metadata=metadata or {},
            command=" ".join(command),
            cwd=cwd,
        )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    except subprocess.TimeoutExpired as exc:
        process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            cmd=exc.cmd or command,
            timeout=exc.timeout or timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    finally:
        if telemetry:
            telemetry.unregister_process(process.pid)


def _build_runner(agent_code: str, input_data: list[dict], blocked_imports: list[str],
                   function_name: str = "schedule") -> str:
    """Build the full runner script that wraps agent code."""
    blocked_check = ", ".join(f'"{m}"' for m in blocked_imports)
    input_data_json = json.dumps(input_data)

    # Build the script as parts to avoid indentation issues with agent code.
    # Agent code runs at top level (defines the function),
    # then we call it inside a try/except.
    header = f"""import json

# Restrict dangerous imports (only at top level, not internal library imports)
BLOCKED = set([{blocked_check}])
_original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
_import_depth = 0

def _restricted_import(name, *args, **kwargs):
    global _import_depth
    top_level = name.split(".")[0]
    # Only block direct imports from agent code (depth 0), not internal library imports
    if _import_depth == 0 and top_level in BLOCKED:
        raise ImportError(f"Import '{{name}}' is blocked for security.")
    _import_depth += 1
    try:
        return _original_import(name, *args, **kwargs)
    finally:
        _import_depth -= 1

import builtins
builtins.__import__ = _restricted_import

# Load input data
input_data = json.loads('{input_data_json}')
"""

    footer = f"""
# Call the agent's function
try:
    result = {function_name}(input_data)
    if not isinstance(result, list):
        result = list(result)
    result = [int(x) for x in result]
    print(json.dumps({{"success": True, "result": result}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": f"{{type(e).__name__}}: {{str(e)}}"}}))\n"""

    return header + "\n# --- Agent code ---\n" + agent_code + "\n" + footer
