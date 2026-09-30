"""Unit tests for lean_build."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lean_lsp_mcp.server import lsp_build


class _FailingClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        raise PermissionError("operation not permitted")


def _make_project(root):
    root.mkdir()
    (root / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    (root / "lakefile.toml").write_text('name = "test"\n')
    return root


@pytest.fixture
def build_mocks(tmp_path):
    """Shared mocks for lsp_build tests."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    (project / "lakefile.toml").write_text('name = "test"\n')

    ctx = MagicMock()
    ctx.request_context.lifespan_context.lean_project_path = project
    ctx.request_context.lifespan_context.client = None
    ctx.request_context.lifespan_context.build_coordinator = None
    ctx.info = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.report_progress = AsyncMock()

    # Simple process for cache (no stdout needed)
    cache_proc = MagicMock()
    cache_proc.wait = AsyncMock()
    cache_proc.stdout.read = AsyncMock(return_value=b"")

    # Build process with stdout
    build_proc = MagicMock()
    build_proc.returncode = 0
    build_proc.wait = AsyncMock()

    return project, ctx, cache_proc, build_proc


def make_read(output: bytes):
    """Create async read that streams output in requested chunk sizes."""
    remaining = output

    async def read(size: int):
        nonlocal remaining
        if not remaining:
            return b""

        chunk = remaining[:size]
        remaining = remaining[size:]
        return chunk

    return read


@pytest.fixture
def patch_build():
    """Context manager to patch all build dependencies."""
    client_cls = MagicMock()
    client_cls.return_value.start = AsyncMock()
    with (
        patch("lean_lsp_mcp.build_utils.asyncio.create_subprocess_exec") as mock_exec,
        patch("lean_lsp_mcp.build_utils.AsyncLeanLSPClient", client_cls),
    ):
        yield mock_exec


@pytest.mark.asyncio
async def test_progress_parsing(build_mocks, patch_build, tmp_path):
    """Lake counts remain visible while request progress strictly increases."""
    project, ctx, _cache_proc, build_proc = build_mocks
    progress_calls = []
    ctx.report_progress = AsyncMock(
        side_effect=lambda progress, total, message: progress_calls.append(
            (progress, total, message)
        )
    )

    build_proc.stdout.read = make_read(
        b"[0/8] Ran job\n[1/8] Built A\n[1/8] Built A\n"
        b"[2/10] Built B\n[0/2] Ran new phase\n"
    )
    patch_build.side_effect = [build_proc]

    await lsp_build(ctx, lean_project_path=str(project))

    assert [p for p, _, _ in progress_calls] == [1, 2, 3, 4, 5]
    assert all(t is None for _, t, _ in progress_calls)
    assert [m for _, _, m in progress_calls] == [
        "[0/8] Ran job",
        "[1/8] Built A",
        "[1/8] Built A",
        "[2/10] Built B",
        "[0/2] Ran new phase",
    ]


@pytest.mark.parametrize("clean", [False, True])
@pytest.mark.parametrize("fetch_cache", [False, True])
async def test_progress_spans_all_commands(
    build_mocks, patch_build, clean, fetch_cache
):
    project, ctx, _, build_proc = build_mocks
    processes = []
    for enabled in (clean, fetch_cache):
        if enabled:
            process = MagicMock()
            process.stdout.read = make_read(b"[0/2] Ran setup\n[1/2] Built setup\n")
            process.wait = AsyncMock()
            processes.append(process)
    build_proc.stdout.read = make_read(b"[0/10] Ran build\n[1/12] Built module\n")
    processes.append(build_proc)
    patch_build.side_effect = processes

    result = await lsp_build(
        ctx, lean_project_path=str(project), clean=clean, fetch_cache=fetch_cache
    )

    calls = ctx.report_progress.await_args_list
    assert result.success
    assert [call.kwargs["progress"] for call in calls] == list(range(1, len(calls) + 1))
    assert all(call.kwargs["total"] is None for call in calls)
    assert "[1/12] Built module" in result.output


async def test_notification_failure_does_not_abort_build(build_mocks, patch_build):
    project, ctx, _, build_proc = build_mocks
    ctx.report_progress.side_effect = RuntimeError("notification unavailable")
    build_proc.stdout.read = make_read(b"[0/2] Built A\n[0/1] Built B\n")
    patch_build.side_effect = [build_proc]

    result = await lsp_build(ctx, lean_project_path=str(project))

    assert result.success
    assert [
        call.kwargs["progress"] for call in ctx.report_progress.await_args_list
    ] == [1, 2]


async def test_failed_build_preserves_progress_and_errors(build_mocks, patch_build):
    project, ctx, _, build_proc = build_mocks
    build_proc.returncode = 1
    build_proc.stdout.read = make_read(b"[0/2] Built A\nerror: failed B\n")
    patch_build.side_effect = [build_proc]

    result = await lsp_build(ctx, lean_project_path=str(project))

    assert not result.success
    assert result.errors == ["error: failed B"]
    assert ctx.report_progress.await_args.kwargs["progress"] == 1


async def test_cancellation_during_progress_stops_build(build_mocks, patch_build):
    project, ctx, _, build_proc = build_mocks
    build_proc.returncode = None
    build_proc.stdout.read = make_read(b"[0/2] Built A\n[1/2] Built B\n")
    ctx.report_progress.side_effect = asyncio.CancelledError()
    patch_build.side_effect = [build_proc]

    with pytest.raises(asyncio.CancelledError):
        await lsp_build(ctx, lean_project_path=str(project))

    build_proc.terminate.assert_called_once()
    build_proc.wait.assert_awaited_once()
    assert ctx.report_progress.await_count == 1


@pytest.mark.asyncio
async def test_filters_trace_lines(build_mocks, patch_build, tmp_path):
    """Verbose trace: and LEAN_PATH= lines are filtered from output."""
    project, ctx, _cache_proc, build_proc = build_mocks
    build_proc.stdout.read = make_read(
        b"[0/2] Built A\ntrace: .> LEAN_PATH=/x lean cmd\n[1/2] Built B\n"
    )
    patch_build.side_effect = [build_proc]

    result = await lsp_build(ctx, lean_project_path=str(project), output_lines=100)

    assert "trace:" not in result.output
    assert "LEAN_PATH" not in result.output
    assert "Built" in result.output


@pytest.mark.asyncio
async def test_output_truncation(build_mocks, patch_build, tmp_path):
    """output_lines parameter truncates to last N lines."""
    project, ctx, _cache_proc, build_proc = build_mocks
    lines = b"\n".join(f"[{i}/50] Built M{i}".encode() for i in range(50))
    build_proc.stdout.read = make_read(lines + b"\nDone\n")
    patch_build.side_effect = [build_proc]

    result = await lsp_build(ctx, lean_project_path=str(project), output_lines=5)

    # Should only have last 5 lines
    assert len(result.output.strip().split("\n")) <= 5


@pytest.mark.asyncio
async def test_output_lines_zero(build_mocks, patch_build, tmp_path):
    """output_lines=0 returns empty output."""
    project, ctx, _cache_proc, build_proc = build_mocks
    build_proc.stdout.read = make_read(b"[0/1] Built\nDone\n")
    patch_build.side_effect = [build_proc]

    result = await lsp_build(ctx, lean_project_path=str(project), output_lines=0)

    assert result.output == ""
    assert result.success


@pytest.mark.asyncio
async def test_default_build_skips_cache_fetch(build_mocks, patch_build, tmp_path):
    """Default build runs lake build without fetching caches."""
    project, ctx, _cache_proc, build_proc = build_mocks
    build_proc.stdout.read = make_read(b"Done\n")
    patch_build.side_effect = [build_proc]

    result = await lsp_build(ctx, lean_project_path=str(project), output_lines=100)

    assert result.success
    assert patch_build.await_count == 1
    assert patch_build.await_args.args[:2] == ("lake", "build")


@pytest.mark.asyncio
async def test_reports_cache_progress(build_mocks, patch_build, tmp_path):
    """Cache fetch is reported via progress."""
    project, ctx, cache_proc, build_proc = build_mocks
    progress_calls = []
    ctx.report_progress = AsyncMock(
        side_effect=lambda progress, total, message: progress_calls.append(
            (progress, total, message)
        )
    )
    build_proc.stdout.read = make_read(b"Done\n")
    patch_build.side_effect = [cache_proc, build_proc]

    await lsp_build(
        ctx, lean_project_path=str(project), fetch_cache=True, output_lines=100
    )

    # Should have reported cache fetch progress
    assert any("cache" in m.lower() for p, t, m in progress_calls)
    assert patch_build.await_args_list[0].args[:4] == (
        "lake",
        "exe",
        "cache",
        "get",
    )


@pytest.mark.asyncio
async def test_setup_subprocesses_pipe_output(build_mocks, patch_build, tmp_path):
    """Setup subprocesses must not inherit stdio."""
    project, ctx, cache_proc, build_proc = build_mocks
    clean_proc = MagicMock()
    clean_proc.wait = AsyncMock()
    clean_proc.stdout.read = AsyncMock(return_value=b"")
    build_proc.stdout.read = make_read(b"Done\n")
    patch_build.side_effect = [clean_proc, cache_proc, build_proc]

    await lsp_build(
        ctx,
        lean_project_path=str(project),
        clean=True,
        fetch_cache=True,
        output_lines=100,
    )

    clean_call = patch_build.await_args_list[0]
    cache_call = patch_build.await_args_list[1]
    assert clean_call.kwargs["stdout"] == asyncio.subprocess.PIPE
    assert clean_call.kwargs["stderr"] == asyncio.subprocess.STDOUT
    assert cache_call.kwargs["stdout"] == asyncio.subprocess.PIPE
    assert cache_call.kwargs["stderr"] == asyncio.subprocess.STDOUT


@pytest.mark.asyncio
async def test_handles_long_verbose_line(build_mocks, patch_build, tmp_path):
    """Long verbose lines do not overflow the stream reader limit."""
    project, ctx, _cache_proc, build_proc = build_mocks
    long_trace = b"trace: " + (b"x" * (70 * 1024)) + b"\n[1/2] Built A\nDone\n"
    build_proc.stdout.read = make_read(long_trace)
    patch_build.side_effect = [build_proc]

    result = await lsp_build(ctx, lean_project_path=str(project), output_lines=100)

    assert result.success
    assert "trace:" not in result.output
    assert "[1/2] Built A" in result.output


@pytest.mark.asyncio
async def test_lsp_build_continues_when_client_close_fails(
    build_mocks, patch_build, tmp_path
):
    """Pre-build client close failure is logged and does not abort the build."""
    project, ctx, _cache_proc, build_proc = build_mocks
    failing_client = _FailingClient()
    failing_client.project_path = tmp_path / "old-proj"
    ctx.request_context.lifespan_context.client = failing_client

    build_proc.stdout.read = make_read(b"[0/1] Built\nDone\n")
    patch_build.side_effect = [build_proc]

    result = await lsp_build(ctx, lean_project_path=str(project), output_lines=100)

    assert failing_client.close_calls == 1
    assert result.success
    assert ctx.request_context.lifespan_context.client is not failing_client
