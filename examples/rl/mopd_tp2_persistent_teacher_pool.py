#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Experimental MOPD teacher pool for TP2 persistent vLLM engines.

This script is intentionally a thin manager around vLLM's existing sleep/wake
HTTP endpoints. It targets the MOPD case where several distinct teacher models
are bound to the same two physical GPUs, but only one teacher is awake on those
GPUs at a time:

  start teacher A/B/C/... as separate vLLM server processes
  initialize them sequentially
  immediately sleep(level=1) each server
  for one teacher prefill:
      acquire an inter-process GPU ownership lock
      wake weights, then wake KV cache + scheduling
      run the prefill/scoring request
      sleep(level=1)
      wait for VRAM to drop before releasing the lock

It does not change vLLM's allocator. The purpose is to make the lifecycle
explicit and reproducible before moving any of this into core vLLM.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class TeacherSpec:
    name: str
    model: str
    port: int
    served_model_name: str


def now() -> float:
    return time.perf_counter()


def wall() -> float:
    return time.time()


class JsonlLogger:
    def __init__(self, path: Path | None):
        self.path = path
        self._fh = path.open("a", encoding="utf-8") if path else None

    def emit(self, event: str, **fields: Any) -> None:
        row = {"event": event, "time": wall(), "perf": now(), **fields}
        line = json.dumps(row, sort_keys=True)
        print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> Any:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        if not data:
            return None
        return json.loads(data.decode("utf-8"))


def post_empty(url: str, params: dict[str, Any] | list[tuple[str, Any]], timeout=600):
    query = urllib.parse.urlencode(params, doseq=True)
    full_url = f"{url}?{query}" if query else url
    req = urllib.request.Request(full_url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def wait_health(base_url: str, deadline_s: float, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + deadline_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {base_url}/health: {last_error}")


def gpu_used_mib(gpus: str) -> dict[str, int]:
    ids = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    cmd = [
        "nvidia-smi",
        f"--id={','.join(ids)}",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    result: dict[str, int] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        idx, used = [part.strip() for part in line.split(",", 1)]
        result[idx] = int(used)
    return result


@contextlib.contextmanager
def exclusive_gpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class PersistentTeacher:
    def __init__(
        self,
        spec: TeacherSpec,
        gpus: str,
        kv_cache_memory_bytes: int | None,
        gpu_memory_utilization: float,
        logger: JsonlLogger,
        extra_vllm_args: list[str],
        server_log_dir: Path | None = None,
    ):
        self.spec = spec
        self.gpus = gpus
        self.kv_cache_memory_bytes = kv_cache_memory_bytes
        self.gpu_memory_utilization = gpu_memory_utilization
        self.logger = logger
        self.extra_vllm_args = extra_vllm_args
        self.server_log_dir = server_log_dir
        self._server_log_fh = None
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.spec.port}"

    def start(self) -> None:
        env = os.environ.copy()
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        env.setdefault("VLLM_SERVER_DEV_MODE", "1")

        cmd = [
            "vllm",
            "serve",
            self.spec.model,
            "--served-model-name",
            self.spec.served_model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.spec.port),
            "--tensor-parallel-size",
            "2",
            "--device-ids",
            self.gpus,
            "--enable-sleep-mode",
            "--enforce-eager",
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]
        if self.kv_cache_memory_bytes is not None:
            cmd += ["--kv-cache-memory-bytes", str(self.kv_cache_memory_bytes)]
        cmd += self.extra_vllm_args

        self.logger.emit(
            "teacher_start",
            teacher=self.spec.name,
            model=self.spec.model,
            port=self.spec.port,
            gpus=self.gpus,
            cmd=cmd,
        )
        stdout = subprocess.DEVNULL
        if self.server_log_dir is not None:
            self.server_log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.server_log_dir / f"{self.spec.name}.vllm.log"
            self._server_log_fh = log_path.open("a", encoding="utf-8")
            stdout = self._server_log_fh
            self.logger.emit(
                "teacher_server_log",
                teacher=self.spec.name,
                path=str(log_path),
            )

        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_health(self.base_url, deadline_s=1800, proc=self.proc)
        self.logger.emit("teacher_healthy", teacher=self.spec.name)

    def sleep(self, level: int = 1) -> None:
        start = now()
        self.logger.emit("teacher_sleep_start", teacher=self.spec.name, level=level)
        post_empty(f"{self.base_url}/sleep", {"level": level, "mode": "abort"})
        self.logger.emit(
            "teacher_sleep_end",
            teacher=self.spec.name,
            level=level,
            duration_s=now() - start,
        )

    def wake(self) -> None:
        start = now()
        self.logger.emit("teacher_wake_weights_start", teacher=self.spec.name)
        post_empty(f"{self.base_url}/wake_up", [("tags", "weights")])
        self.logger.emit(
            "teacher_wake_weights_end",
            teacher=self.spec.name,
            duration_s=now() - start,
        )

        start = now()
        self.logger.emit("teacher_wake_kv_start", teacher=self.spec.name)
        post_empty(
            f"{self.base_url}/wake_up",
            [("tags", "kv_cache"), ("tags", "scheduling")],
        )
        self.logger.emit(
            "teacher_wake_kv_end",
            teacher=self.spec.name,
            duration_s=now() - start,
        )

    def completion(self, prompts: list[str], prompt_logprobs: int) -> Any:
        payload = {
            "model": self.spec.served_model_name,
            "prompt": prompts,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": prompt_logprobs,
        }
        start = now()
        self.logger.emit(
            "teacher_prefill_start",
            teacher=self.spec.name,
            num_prompts=len(prompts),
            prompt_logprobs=prompt_logprobs,
        )
        result = http_json("POST", f"{self.base_url}/v1/completions", payload)
        self.logger.emit(
            "teacher_prefill_end",
            teacher=self.spec.name,
            duration_s=now() - start,
        )
        return result

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.logger.emit("teacher_stop", teacher=self.spec.name)
        os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=30)
        if self._server_log_fh is not None:
            self._server_log_fh.close()
            self._server_log_fh = None


class PersistentTeacherPool:
    def __init__(
        self,
        teachers: list[PersistentTeacher],
        lock_file: Path,
        gpus: str,
        sleep_used_mib_threshold: int | None,
        logger: JsonlLogger,
    ):
        self.teachers = {teacher.spec.name: teacher for teacher in teachers}
        self.lock_file = lock_file
        self.gpus = gpus
        self.sleep_used_mib_threshold = sleep_used_mib_threshold
        self.logger = logger

    def start_all_sleeping(self) -> None:
        for teacher in self.teachers.values():
            teacher.start()
            teacher.sleep(level=1)
            self._wait_for_sleep_vram(teacher.spec.name)

    def prefill(self, teacher_name: str, prompts: list[str], prompt_logprobs: int):
        teacher = self.teachers[teacher_name]
        with exclusive_gpu_lock(self.lock_file):
            self.logger.emit("gpu_lock_acquired", teacher=teacher_name)
            teacher.wake()
            try:
                return teacher.completion(prompts, prompt_logprobs)
            finally:
                teacher.sleep(level=1)
                self._wait_for_sleep_vram(teacher_name)
                self.logger.emit("gpu_lock_released", teacher=teacher_name)

    def _wait_for_sleep_vram(self, teacher_name: str) -> None:
        if self.sleep_used_mib_threshold is None:
            return
        start = now()
        while True:
            used = gpu_used_mib(self.gpus)
            max_used = max(used.values()) if used else 0
            self.logger.emit(
                "sleep_vram_poll",
                teacher=teacher_name,
                max_used_mib=max_used,
                per_gpu_used_mib=used,
            )
            if max_used <= self.sleep_used_mib_threshold:
                self.logger.emit(
                    "sleep_vram_ready",
                    teacher=teacher_name,
                    duration_s=now() - start,
                    max_used_mib=max_used,
                )
                return
            time.sleep(1)

    def stop_all(self) -> None:
        for teacher in self.teachers.values():
            teacher.stop()


def parse_teacher(raw: str, port: int) -> TeacherSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "--teacher must have the form NAME=MODEL_OR_PATH"
        )
    name, model = raw.split("=", 1)
    name = name.strip()
    model = model.strip()
    if not name or not model:
        raise argparse.ArgumentTypeError(
            "--teacher must have non-empty NAME and MODEL_OR_PATH"
        )
    return TeacherSpec(
        name=name,
        model=model,
        port=port,
        served_model_name=f"teacher-{name}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", action="append", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--base-port", type=int, default=19080)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--sleep-used-mib-threshold", type=int)
    parser.add_argument("--lock-file", type=Path, default=Path("/tmp/vllm_tp2_pool.lock"))
    parser.add_argument("--log-jsonl", type=Path)
    parser.add_argument("--server-log-dir", type=Path)
    parser.add_argument("--prompt-logprobs", type=int, default=64)
    parser.add_argument("--smoke-prompt", action="append")
    parser.add_argument(
        "--exit-after-smoke",
        action="store_true",
        help="Exit after optional smoke prefills instead of keeping servers alive.",
    )
    parser.add_argument(
        "--extra-vllm-arg",
        action="append",
        default=[],
        help="Additional raw arguments appended to each `vllm serve` command.",
    )
    args = parser.parse_args()

    logger = JsonlLogger(args.log_jsonl)
    teachers = [
        PersistentTeacher(
            parse_teacher(raw, args.base_port + idx),
            gpus=args.gpus,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            gpu_memory_utilization=args.gpu_memory_utilization,
            logger=logger,
            extra_vllm_args=args.extra_vllm_arg,
            server_log_dir=args.server_log_dir,
        )
        for idx, raw in enumerate(args.teacher)
    ]
    pool = PersistentTeacherPool(
        teachers,
        lock_file=args.lock_file,
        gpus=args.gpus,
        sleep_used_mib_threshold=args.sleep_used_mib_threshold,
        logger=logger,
    )

    try:
        pool.start_all_sleeping()
        if args.smoke_prompt:
            for teacher in teachers:
                pool.prefill(
                    teacher.spec.name,
                    prompts=args.smoke_prompt,
                    prompt_logprobs=args.prompt_logprobs,
                )
        logger.emit("teacher_pool_ready", teachers=list(pool.teachers))
        if not args.exit_after_smoke:
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        logger.emit("teacher_pool_interrupted")
    finally:
        pool.stop_all()
        logger.close()


if __name__ == "__main__":
    main()
