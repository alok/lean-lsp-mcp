from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lean_lsp_mcp.models import BuildResult
from lean_lsp_mcp.server import BuildCoordinator
from lean_lsp_mcp.build_utils import LakeBuildRunner, run_build


@pytest.mark.parametrize("mode", ["cancel", "share"])
async def test_supersession_chain_waits_for_cleanup(mode: str) -> None:
    coordinator = BuildCoordinator(mode)
    started, cleaning, release = (asyncio.Event() for _ in range(3))
    replacements_started: list[int] = []

    async def first() -> BuildResult:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
        return BuildResult(success=True, output="first", errors=[])

    async def replacement(index: int) -> BuildResult:
        replacements_started.append(index)
        return BuildResult(success=True, output=str(index), errors=[])

    callers = [asyncio.create_task(coordinator.run(first))]
    await started.wait()
    # The middle build can be cancelled before its coroutine starts.
    callers.extend(
        asyncio.create_task(coordinator.run(lambda i=i: replacement(i)))
        for i in range(2)
    )
    await cleaning.wait()
    for _ in range(5):
        await asyncio.sleep(0)
    try:
        assert not replacements_started
    finally:
        release.set()
        results = await asyncio.gather(*callers)
    assert replacements_started == [1]
    assert results[-1].output == "1"
    assert not coordinator._waiters


@pytest.mark.parametrize("mode", ["cancel", "share"])
async def test_repeated_request_cancellation_does_not_interrupt_cleanup(
    mode: str,
) -> None:
    coordinator = BuildCoordinator(mode)
    started, cleaning, release, cleaned = (asyncio.Event() for _ in range(4))

    async def build() -> BuildResult:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            cleaned.set()
        return BuildResult(success=True, output="", errors=[])

    caller = asyncio.create_task(coordinator.run(build))
    await started.wait()
    caller.cancel()
    await cleaning.wait()
    caller.cancel()
    await asyncio.sleep(0)
    try:
        assert not caller.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
    assert cleaned.is_set()
    assert not coordinator._waiters


@pytest.mark.parametrize("mode", ["cancel", "share"])
async def test_failed_cleanup_does_not_block_next_build(mode: str) -> None:
    coordinator = BuildCoordinator(mode)
    started = asyncio.Event()

    async def first() -> BuildResult:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            raise RuntimeError("cleanup failed")

    async def replacement() -> BuildResult:
        return BuildResult(success=True, output="replacement", errors=[])

    caller = asyncio.create_task(coordinator.run(first))
    await started.wait()
    result = await coordinator.run(replacement)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await caller
    assert result.success
    assert not coordinator._waiters


@pytest.mark.parametrize("mode", ["allow", "cancel", "share"])
async def test_cancelled_only_caller_waits_for_build_cleanup(mode: str) -> None:
    coordinator = BuildCoordinator(mode)
    started, cleaning, release, cleaned = (asyncio.Event() for _ in range(4))

    async def build() -> BuildResult:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            cleaned.set()
        return BuildResult(success=True, output="", errors=[])

    caller = asyncio.create_task(coordinator.run(build))
    await started.wait()
    caller.cancel()
    try:
        await asyncio.wait_for(cleaning.wait(), timeout=0.5)
        assert not caller.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert cleaned.is_set()
    finally:
        release.set()
        if coordinator._current is not None:
            coordinator._current.cancel()
            await asyncio.gather(coordinator._current.task, return_exceptions=True)
        await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.parametrize("cancel_latest", [False, True])
async def test_cancelled_share_waiter_preserves_other_waiter(
    cancel_latest: bool,
) -> None:
    coordinator = BuildCoordinator("share")
    started, latest_started, finish, cancelled = (asyncio.Event() for _ in range(4))

    async def first() -> BuildResult:
        started.set()
        await asyncio.Event().wait()
        return BuildResult(success=True, output="first", errors=[])

    async def latest() -> BuildResult:
        latest_started.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return BuildResult(success=True, output="latest", errors=[])

    callers = [asyncio.create_task(coordinator.run(first))]
    await started.wait()
    callers.append(asyncio.create_task(coordinator.run(latest)))
    await latest_started.wait()
    # Let the first caller adopt the latest build before cancelling it.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    leaving = callers[int(cancel_latest)]
    remaining = callers[int(not cancel_latest)]
    leaving.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await leaving
        assert not cancelled.is_set()
        finish.set()
        assert (await remaining).output == "latest"
    finally:
        finish.set()
        await asyncio.gather(*callers, return_exceptions=True)


@pytest.mark.parametrize("mode", ["cancel", "share"])
async def test_superseding_build_waits_for_previous_cleanup(mode: str) -> None:
    coordinator = BuildCoordinator(mode)
    started, cleaning, release, next_started = (asyncio.Event() for _ in range(4))

    async def first() -> BuildResult:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
        return BuildResult(success=True, output="first", errors=[])

    async def second() -> BuildResult:
        next_started.set()
        return BuildResult(success=True, output="second", errors=[])

    caller = asyncio.create_task(coordinator.run(first))
    await started.wait()
    replacement = asyncio.create_task(coordinator.run(second))
    await cleaning.wait()
    try:
        assert not next_started.is_set()
    finally:
        release.set()
        await asyncio.gather(caller, replacement, return_exceptions=True)


@pytest.mark.parametrize("mode", ["cancel", "share"])
async def test_cancelled_replacement_during_cleanup(mode: str) -> None:
    coordinator = BuildCoordinator(mode)
    started, cleaning, release, next_started = (asyncio.Event() for _ in range(4))

    async def first() -> BuildResult:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
        return BuildResult(success=True, output="first", errors=[])

    async def second() -> BuildResult:
        next_started.set()
        return BuildResult(success=True, output="second", errors=[])

    caller = asyncio.create_task(coordinator.run(first))
    await started.wait()
    replacement = asyncio.create_task(coordinator.run(second))
    await cleaning.wait()
    replacement.cancel()
    release.set()
    try:
        with pytest.raises(asyncio.CancelledError):
            await replacement
        result = await caller
        assert next_started.is_set() == (mode == "share")
        assert result.success == (mode == "share")
    finally:
        await asyncio.gather(caller, replacement, return_exceptions=True)


@pytest.mark.parametrize("mode", ["cancel", "share"])
async def test_last_caller_cancellation_reaps_direct_process(
    mode: str, tmp_path: Path
) -> None:
    coordinator = BuildCoordinator(mode)
    ctx = MagicMock()
    ctx.request_context.lifespan_context = SimpleNamespace(client=None)
    ctx.report_progress = AsyncMock()
    started = asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []
    create_process = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs) -> asyncio.subprocess.Process:
        process = await create_process(
            sys.executable,
            "-c",
            "import time; print('ready', flush=True); time.sleep(60)",
            **kwargs,
        )
        processes.append(process)
        started.set()
        return process

    with patch("lean_lsp_mcp.build_utils.asyncio.create_subprocess_exec", spawn):
        caller = asyncio.create_task(
            coordinator.run(lambda: run_build(ctx, tmp_path, False, False, 20))
        )
        await started.wait()
        caller.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, timeout=2)
            assert processes[0].returncode is not None
        finally:
            if processes[0].returncode is None:
                processes[0].kill()
                await processes[0].wait()


async def test_runner_kills_process_that_ignores_termination(tmp_path: Path) -> None:
    runner = LakeBuildRunner(MagicMock())
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
        cwd=tmp_path,
    )
    assert process.stdout is not None
    await process.stdout.readline()
    runner.active_process = process
    wait_for = asyncio.wait_for

    async def short_wait(awaitable, timeout):
        return await wait_for(awaitable, timeout=0.05)

    try:
        with patch("lean_lsp_mcp.build_utils.asyncio.wait_for", short_wait):
            await runner.cancel()
        assert process.returncode == -9
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_build_coordinator_cancel_mode() -> None:
    coordinator = BuildCoordinator("cancel")
    started = asyncio.Event()

    async def build_one() -> BuildResult:
        started.set()
        await asyncio.Event().wait()
        return BuildResult(success=True, output="one", errors=[])

    async def build_two() -> BuildResult:
        return BuildResult(success=True, output="two", errors=[])

    task_one = asyncio.create_task(coordinator.run(build_one))
    await started.wait()
    task_two = asyncio.create_task(coordinator.run(build_two))

    result_two = await task_two
    result_one = await task_one

    assert result_two.output == "two"
    assert result_one.success is False
    assert result_one.errors and "superseded" in result_one.errors[0].lower()


@pytest.mark.asyncio
async def test_build_coordinator_share_mode() -> None:
    coordinator = BuildCoordinator("share")
    started = asyncio.Event()

    async def build_one() -> BuildResult:
        started.set()
        await asyncio.Event().wait()
        return BuildResult(success=True, output="one", errors=[])

    async def build_two() -> BuildResult:
        return BuildResult(success=True, output="two", errors=[])

    task_one = asyncio.create_task(coordinator.run(build_one))
    await started.wait()
    task_two = asyncio.create_task(coordinator.run(build_two))

    result_one = await task_one
    result_two = await task_two

    assert result_one.output == "two"
    assert result_two.output == "two"


@pytest.mark.asyncio
async def test_build_coordinator_cancel_mode_many_callers_deterministic() -> None:
    coordinator = BuildCoordinator("cancel")
    caller_count = 5
    tasks: list[asyncio.Task[BuildResult]] = []

    for i in range(caller_count):
        started = asyncio.Event()

        if i < caller_count - 1:

            async def build(
                i: int = i, started: asyncio.Event = started
            ) -> BuildResult:
                started.set()
                await asyncio.Event().wait()
                return BuildResult(success=True, output=f"build-{i}", errors=[])

        else:

            async def build(
                i: int = i, started: asyncio.Event = started
            ) -> BuildResult:
                started.set()
                await asyncio.sleep(0)
                return BuildResult(success=True, output=f"build-{i}", errors=[])

        tasks.append(asyncio.create_task(coordinator.run(build)))
        await started.wait()

    results = await asyncio.gather(*tasks)
    final_output = f"build-{caller_count - 1}"
    superseded = [result for result in results if not result.success]
    succeeded = [result for result in results if result.success]

    assert len(superseded) == caller_count - 1
    assert len(succeeded) == 1
    assert succeeded[0].output == final_output
    assert all(
        result.errors and "superseded" in result.errors[0].lower()
        for result in superseded
    )


@pytest.mark.asyncio
async def test_build_coordinator_share_mode_many_callers_get_latest_result() -> None:
    coordinator = BuildCoordinator("share")
    caller_count = 5
    tasks: list[asyncio.Task[BuildResult]] = []
    final_output = f"build-{caller_count - 1}"

    for i in range(caller_count):
        started = asyncio.Event()

        if i < caller_count - 1:

            async def build(
                i: int = i, started: asyncio.Event = started
            ) -> BuildResult:
                started.set()
                await asyncio.Event().wait()
                return BuildResult(success=True, output=f"build-{i}", errors=[])

        else:

            async def build(
                i: int = i, started: asyncio.Event = started
            ) -> BuildResult:
                started.set()
                await asyncio.sleep(0)
                return BuildResult(success=True, output=f"build-{i}", errors=[])

        tasks.append(asyncio.create_task(coordinator.run(build)))
        await started.wait()

    results = await asyncio.gather(*tasks)
    assert all(result.success for result in results)
    assert all(result.output == final_output for result in results)
