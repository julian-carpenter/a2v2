"""Pure SLURM, topology, locking, and preemption runtime contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import tempfile
import fcntl
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


class TopologyError(ValueError):
    """Raised when scheduler or torchrun topology cannot be mapped safely."""


def _required_text(environment: Mapping[str, str], name: str) -> str:
    """Read one nonempty environment variable or name it in the error."""

    value = environment.get(name)
    if value is None or not value.strip():
        raise TopologyError(f"missing required environment variable {name}")
    return value


def _integer_environment(
    environment: Mapping[str, str],
    name: str,
    *,
    minimum: int,
) -> int:
    """Parse one bounded base-ten scheduler integer."""

    raw = _required_text(environment, name)
    try:
        value = int(raw)
    except ValueError as error:
        raise TopologyError(f"{name} must be a base-ten integer, received {raw!r}") from error
    if value < minimum:
        raise TopologyError(f"{name} must be at least {minimum}, received {value}")
    return value


@dataclass(frozen=True)
class SlurmEnvironment:
    """Validated homogeneous allocation values provided to one node agent."""

    job_id: str
    node_count: int
    node_id: int
    processes_per_node: int
    node_list: str

    @property
    def world_size(self) -> int:
        """Return the homogeneous global worker count."""

        return self.node_count * self.processes_per_node

    @classmethod
    def from_mapping(cls, environment: Mapping[str, str]) -> SlurmEnvironment:
        """Validate the scheduler fields needed by the node launcher."""

        job_id = _required_text(environment, "SLURM_JOB_ID")
        node_count = _integer_environment(
            environment,
            "SLURM_JOB_NUM_NODES",
            minimum=1,
        )
        node_id = _integer_environment(environment, "SLURM_NODEID", minimum=0)
        processes_per_node = _integer_environment(
            environment,
            "SLURM_GPUS_ON_NODE",
            minimum=1,
        )
        node_list = _required_text(environment, "SLURM_JOB_NODELIST")
        if node_id >= node_count:
            raise TopologyError(
                "SLURM_NODEID must be smaller than SLURM_JOB_NUM_NODES; "
                f"received {node_id} for {node_count} nodes"
            )
        return cls(
            job_id=job_id,
            node_count=node_count,
            node_id=node_id,
            processes_per_node=processes_per_node,
            node_list=node_list,
        )


def homogeneous_world_size(
    processes_by_node: Sequence[int],
    *,
    node_count: int,
) -> int:
    """Validate one positive, equal process count for every allocated node."""

    if len(processes_by_node) != node_count:
        raise TopologyError(
            f"expected process counts for {node_count} nodes, received "
            f"{len(processes_by_node)}"
        )
    if not processes_by_node or any(
        type(count) is not int or count <= 0 for count in processes_by_node
    ):
        raise TopologyError("per-node process counts must be positive integers")
    if len(set(processes_by_node)) != 1:
        raise TopologyError(
            "SLURM launch requires homogeneous per-node process counts; "
            f"received {list(processes_by_node)}"
        )
    return node_count * processes_by_node[0]


@dataclass(frozen=True)
class Rendezvous:
    """The c10d endpoint and phase-unique rendezvous identifier."""

    endpoint: str
    rendezvous_id: str


_PHASE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENDPOINT_PATTERN = re.compile(r"^(?:\[[^\]]+\]|[^:\s]+):([0-9]+)$")


def _validate_endpoint(endpoint: str) -> str:
    """Return a host:port endpoint after structural and port validation."""

    match = _ENDPOINT_PATTERN.fullmatch(endpoint)
    if match is None:
        raise TopologyError(
            f"rendezvous endpoint must use host:port syntax, received {endpoint!r}"
        )
    port = int(match.group(1))
    if not 1 <= port <= 65535:
        raise TopologyError(f"rendezvous port must be in 1..65535, received {port}")
    return endpoint


def build_rendezvous(
    allocation: SlurmEnvironment,
    *,
    phase: str,
    port: int,
    resolve_hosts: Callable[[str], Sequence[str]],
    explicit_endpoint: str | None = None,
) -> Rendezvous:
    """Build one phase rendezvous without running a scheduler command."""

    if _PHASE_PATTERN.fullmatch(phase) is None:
        raise TopologyError(
            "phase must contain only letters, digits, dots, underscores, and dashes"
        )
    if explicit_endpoint is None:
        if not 1 <= port <= 65535:
            raise TopologyError(f"rendezvous port must be in 1..65535, received {port}")
        hosts = tuple(resolve_hosts(allocation.node_list))
        if not hosts or not hosts[0].strip():
            raise TopologyError(
                f"could not resolve the first host in {allocation.node_list!r}"
            )
        endpoint = f"{hosts[0]}:{port}"
    else:
        endpoint = explicit_endpoint
    return Rendezvous(
        endpoint=_validate_endpoint(endpoint),
        rendezvous_id=f"{allocation.job_id}-{phase}",
    )


@dataclass(frozen=True)
class RunContract:
    """Fields that every node agent must agree on before training starts."""

    job_id: str
    phase: str
    node_count: int
    processes_per_node: int
    world_size: int
    rendezvous_endpoint: str
    rendezvous_id: str
    output_directory: str
    manifest_fingerprint: str

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical JSON-safe contract mapping."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RunContract:
        """Parse an exact run-contract mapping."""

        required = tuple(cls.__dataclass_fields__)
        missing = sorted(set(required) - set(value))
        extra = sorted(set(value) - set(required))
        if missing or extra:
            raise TopologyError(
                f"run contract keys differ: missing={missing}, extra={extra}"
            )
        return cls(**{key: value[key] for key in required})  # type: ignore[arg-type]

    def fingerprint(self) -> str:
        """Hash the canonical serialized contract for cross-node comparison."""

        encoded = json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def contract_differences(
    expected: RunContract | Mapping[str, Any],
    actual: RunContract | Mapping[str, Any],
) -> dict[str, tuple[object, object]]:
    """Return each mismatching contract field in deterministic order."""

    expected_values = (
        expected.to_mapping() if isinstance(expected, RunContract) else dict(expected)
    )
    actual_values = actual.to_mapping() if isinstance(actual, RunContract) else dict(actual)
    return {
        key: (expected_values.get(key), actual_values.get(key))
        for key in sorted(set(expected_values) | set(actual_values))
        if expected_values.get(key) != actual_values.get(key)
    }


_TORCHRUN_VARIABLES = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")


@dataclass(frozen=True)
class DistributedEnvironment:
    """Validated global and node-local ranks supplied by torchrun."""

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int

    @property
    def node_rank(self) -> int:
        """Derive the homogeneous node rank from the global worker rank."""

        return self.rank // self.local_world_size

    @classmethod
    def from_mapping(cls, environment: Mapping[str, str]) -> DistributedEnvironment:
        """Preserve local defaults or require one complete torchrun topology."""

        present = [name for name in _TORCHRUN_VARIABLES if name in environment]
        if not present:
            return cls(rank=0, local_rank=0, world_size=1, local_world_size=1)
        missing = [name for name in _TORCHRUN_VARIABLES if name not in environment]
        if missing:
            raise TopologyError(
                "partial torchrun environment is missing required variables: "
                f"{missing}"
            )
        rank = _integer_environment(environment, "RANK", minimum=0)
        local_rank = _integer_environment(environment, "LOCAL_RANK", minimum=0)
        world_size = _integer_environment(environment, "WORLD_SIZE", minimum=1)
        local_world_size = _integer_environment(
            environment,
            "LOCAL_WORLD_SIZE",
            minimum=1,
        )
        if rank >= world_size:
            raise TopologyError(
                f"RANK must be smaller than WORLD_SIZE; received {rank} for {world_size}"
            )
        if local_rank >= local_world_size:
            raise TopologyError(
                "LOCAL_RANK must be smaller than LOCAL_WORLD_SIZE; "
                f"received {local_rank} for {local_world_size}"
            )
        if world_size % local_world_size != 0:
            raise TopologyError(
                "WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE for a homogeneous "
                f"launch; received {world_size} and {local_world_size}"
            )
        if rank % local_world_size != local_rank:
            raise TopologyError(
                "RANK and LOCAL_RANK disagree for LOCAL_WORLD_SIZE: "
                f"{rank} % {local_world_size} != {local_rank}"
            )
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
        )


def validate_runtime_topology(
    distributed: DistributedEnvironment,
    *,
    slurm: SlurmEnvironment | None,
    cuda_requested: bool,
    visible_cuda_devices: int,
) -> None:
    """Fail before process-group setup when ranks cannot map to the allocation."""

    if type(visible_cuda_devices) is not int or visible_cuda_devices < 0:
        raise TopologyError("visible CUDA device count must be a nonnegative integer")
    if slurm is not None:
        if distributed.world_size != slurm.world_size:
            raise TopologyError(
                f"WORLD_SIZE={distributed.world_size} does not match SLURM allocation "
                f"world size {slurm.world_size}"
            )
        if distributed.local_world_size != slurm.processes_per_node:
            raise TopologyError(
                f"LOCAL_WORLD_SIZE={distributed.local_world_size} does not match "
                f"SLURM_GPUS_ON_NODE={slurm.processes_per_node}"
            )
        if distributed.node_rank != slurm.node_id:
            raise TopologyError(
                f"derived node rank {distributed.node_rank} does not match "
                f"SLURM_NODEID={slurm.node_id}"
            )
    if not cuda_requested:
        return
    if distributed.local_rank >= visible_cuda_devices:
        raise TopologyError(
            f"LOCAL_RANK={distributed.local_rank} has no visible CUDA device; "
            f"visible CUDA device count is {visible_cuda_devices}"
        )
    if distributed.local_world_size > visible_cuda_devices:
        raise TopologyError(
            f"LOCAL_WORLD_SIZE={distributed.local_world_size} exceeds visible CUDA "
            f"device count {visible_cuda_devices}"
        )
    if slurm is not None and visible_cuda_devices != slurm.processes_per_node:
        raise TopologyError(
            f"visible CUDA device count {visible_cuda_devices} does not match "
            f"SLURM_GPUS_ON_NODE={slurm.processes_per_node}"
        )


TOPOLOGY_SCHEMA = "a2v2.topology.v1"


@dataclass(frozen=True)
class RankTopology:
    """JSON-safe hardware and rank metadata captured for one global rank."""

    hostname: str
    global_rank: int
    local_rank: int
    local_world_size: int
    visible_cuda_devices: int
    selected_cuda_device: int | None
    cuda_device_name: str | None
    cuda_device_uuid: str | None

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-safe record without runtime device objects."""

        return asdict(self)


_RANK_TOPOLOGY_FIELDS = tuple(RankTopology.__dataclass_fields__)


def _rank_topology_from_mapping(value: Mapping[str, object]) -> RankTopology:
    """Validate the exact types and local bounds of one saved rank record."""

    missing = sorted(set(_RANK_TOPOLOGY_FIELDS) - set(value))
    extra = sorted(set(value) - set(_RANK_TOPOLOGY_FIELDS))
    if missing or extra:
        raise TopologyError(
            f"rank topology keys differ: missing={missing}, extra={extra}"
        )
    text_fields = ("hostname",)
    optional_text_fields = ("cuda_device_name", "cuda_device_uuid")
    integer_fields = (
        "global_rank",
        "local_rank",
        "local_world_size",
        "visible_cuda_devices",
    )
    for name in text_fields:
        if not isinstance(value[name], str) or not value[name]:
            raise TopologyError(f"rank topology {name} must be nonempty text")
    for name in optional_text_fields:
        if value[name] is not None and not isinstance(value[name], str):
            raise TopologyError(f"rank topology {name} must be text or null")
    for name in integer_fields:
        if type(value[name]) is not int:
            raise TopologyError(f"rank topology {name} must be an integer")
    selected = value["selected_cuda_device"]
    if selected is not None and type(selected) is not int:
        raise TopologyError(
            "rank topology selected_cuda_device must be an integer or null"
        )
    record = RankTopology(**{  # type: ignore[arg-type]
        name: value[name] for name in _RANK_TOPOLOGY_FIELDS
    })
    if record.global_rank < 0:
        raise TopologyError("rank topology global_rank must be nonnegative")
    if record.local_world_size <= 0:
        raise TopologyError("rank topology local_world_size must be positive")
    if not 0 <= record.local_rank < record.local_world_size:
        raise TopologyError(
            "rank topology local_rank must be within local_world_size"
        )
    if record.visible_cuda_devices < 0:
        raise TopologyError("rank topology visible_cuda_devices must be nonnegative")
    if record.selected_cuda_device is not None and not (
        0 <= record.selected_cuda_device < record.visible_cuda_devices
    ):
        raise TopologyError(
            "rank topology selected_cuda_device must name a visible CUDA device"
        )
    return record


def build_topology_state(records: Sequence[RankTopology]) -> dict[str, object]:
    """Build one rank-indexed topology payload after structural validation."""

    ordered = sorted(records, key=lambda record: record.global_rank)
    expected_ranks = list(range(len(ordered)))
    actual_ranks = [record.global_rank for record in ordered]
    if actual_ranks != expected_ranks:
        raise TopologyError(
            f"topology global ranks must be contiguous {expected_ranks}, "
            f"received {actual_ranks}"
        )
    validated = [
        _rank_topology_from_mapping(record.to_mapping()) for record in ordered
    ]
    return {
        "schema": TOPOLOGY_SCHEMA,
        "world_size": len(validated),
        "by_rank": [record.to_mapping() for record in validated],
    }


def gather_rank_topologies(
    local: RankTopology,
    *,
    world_size: int,
    rank: int,
    group: dist.ProcessGroup | None,
) -> dict[str, object] | None:
    """Gather JSON-safe rank records onto the sole checkpoint-writing rank."""

    if world_size <= 0 or not 0 <= rank < world_size:
        raise TopologyError(
            f"invalid topology gather rank/world size: rank={rank}, world_size={world_size}"
        )
    gathered: list[object] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local.to_mapping(), gathered, dst=0, group=group)
    if rank != 0:
        return None
    if gathered is None or len(gathered) != world_size or any(
        not isinstance(item, Mapping) for item in gathered
    ):
        raise TopologyError(
            "distributed topology gather did not return one mapping per rank"
        )
    records = tuple(_rank_topology_from_mapping(item) for item in gathered)
    return build_topology_state(records)


def _parse_topology_state(state: Mapping[str, object]) -> tuple[RankTopology, ...]:
    """Validate one topology schema and return globally indexed records."""

    expected_keys = {"schema", "world_size", "by_rank"}
    if set(state) != expected_keys:
        raise TopologyError(
            "topology state keys must be exactly schema, world_size, and by_rank"
        )
    if state["schema"] != TOPOLOGY_SCHEMA:
        raise TopologyError(
            f"unsupported topology schema {state['schema']!r}; expected {TOPOLOGY_SCHEMA}"
        )
    world_size = state["world_size"]
    by_rank = state["by_rank"]
    if type(world_size) is not int or world_size <= 0:
        raise TopologyError("topology world_size must be a positive integer")
    if not isinstance(by_rank, list) or len(by_rank) != world_size:
        raise TopologyError(
            "topology by_rank must contain one record per world-size rank"
        )
    records: list[RankTopology] = []
    for value in by_rank:
        if not isinstance(value, Mapping):
            raise TopologyError("topology rank records must be mappings")
        records.append(_rank_topology_from_mapping(value))
    if [record.global_rank for record in records] != list(range(world_size)):
        raise TopologyError("topology records must be indexed by contiguous global ranks")
    return tuple(records)


def validate_resume_topology(
    saved: Mapping[str, object],
    *,
    current: RankTopology,
    world_size: int,
) -> tuple[str, ...]:
    """Reject mathematical topology drift and report hardware differences."""

    records = _parse_topology_state(saved)
    saved_world_size = len(records)
    if saved_world_size != world_size:
        raise TopologyError(
            f"checkpoint topology has world size {saved_world_size}, current launch "
            f"has {world_size}"
        )
    if not 0 <= current.global_rank < saved_world_size:
        raise TopologyError(
            f"current global rank {current.global_rank} is outside checkpoint topology"
        )
    validated_current = _rank_topology_from_mapping(current.to_mapping())
    previous = records[current.global_rank]
    warning_fields = (
        "hostname",
        "local_rank",
        "local_world_size",
        "visible_cuda_devices",
        "selected_cuda_device",
        "cuda_device_name",
        "cuda_device_uuid",
    )
    warnings = []
    for name in warning_fields:
        before = getattr(previous, name)
        after = getattr(validated_current, name)
        if before != after:
            warnings.append(
                f"rank {current.global_rank} topology differs at {name}: "
                f"checkpoint={before!r}, current={after!r}"
            )
    return tuple(warnings)


OUTPUT_LOCK_NAME = ".a2v2-output.lock"
OUTPUT_LOCK_SCHEMA = "a2v2.output-lock.v1"


class OutputLockError(RuntimeError):
    """Raised when a save directory has no safe ownership transition."""


@dataclass(frozen=True)
class LockOwner:
    """The scheduler run and process that exclusively own one save directory."""

    job_id: str
    run_id: str
    hostname: str
    pid: int

    def __post_init__(self) -> None:
        """Reject identities that cannot safely round-trip through JSON."""

        for name in ("job_id", "run_id", "hostname"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise OutputLockError(f"lock owner {name} must be nonempty text")
        if type(self.pid) is not int or self.pid <= 0:
            raise OutputLockError("lock owner pid must be a positive integer")

    def to_mapping(self) -> dict[str, object]:
        """Return the complete on-disk ownership record."""

        return {
            "schema": OUTPUT_LOCK_SCHEMA,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "hostname": self.hostname,
            "pid": self.pid,
        }

    def fingerprint(self) -> str:
        """Hash the exact owner record required by explicit recovery."""

        encoded = json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _parse_lock_owner(value: Mapping[str, object]) -> LockOwner:
    """Validate an exact on-disk output lock schema."""

    expected = {"schema", "job_id", "run_id", "hostname", "pid"}
    if set(value) != expected:
        raise OutputLockError(
            "output lock keys must be exactly schema, job_id, run_id, hostname, and pid"
        )
    if value["schema"] != OUTPUT_LOCK_SCHEMA:
        raise OutputLockError(
            f"unsupported output lock schema {value['schema']!r}"
        )
    try:
        return LockOwner(
            job_id=value["job_id"],  # type: ignore[arg-type]
            run_id=value["run_id"],  # type: ignore[arg-type]
            hostname=value["hostname"],  # type: ignore[arg-type]
            pid=value["pid"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise OutputLockError(f"invalid output lock owner: {error}") from error


def _read_lock_owner(path: Path) -> LockOwner:
    """Read a complete atomic lock record or fail closed."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OutputLockError(f"cannot read output lock {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise OutputLockError("output lock root must be a JSON object")
    return _parse_lock_owner(value)


@contextmanager
def _directory_mutex(directory: Path) -> Iterator[None]:
    """Serialize lock-name transitions on the stable output-directory inode."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass
class OutputLock:
    """An acquired lock tied to the exact file inode and owner record."""

    path: Path
    owner: LockOwner
    _device: int
    _inode: int
    _released: bool = False

    @classmethod
    def acquire(cls, output_directory: str | Path, owner: LockOwner) -> OutputLock:
        """Atomically link a complete candidate record into the lock name."""

        directory = Path(output_directory)
        if not directory.is_dir():
            raise OutputLockError(
                f"output directory must exist before locking: {directory}"
            )
        lock_path = directory / OUTPUT_LOCK_NAME
        candidate_fd, candidate_name = tempfile.mkstemp(
            dir=directory,
            prefix=f"{OUTPUT_LOCK_NAME}.",
            suffix=".candidate",
        )
        candidate_path = Path(candidate_name)
        encoded = (
            json.dumps(owner.to_mapping(), sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        try:
            with os.fdopen(candidate_fd, "wb") as candidate:
                candidate.write(encoded)
                candidate.flush()
                os.fsync(candidate.fileno())
            with _directory_mutex(directory):
                try:
                    os.link(candidate_path, lock_path)
                except FileExistsError as error:
                    existing = _read_lock_owner(lock_path)
                    raise OutputLockError(
                        f"output lock owned by run {existing.run_id!r} (job "
                        f"{existing.job_id!r}) blocks requested run {owner.run_id!r} "
                        f"(job {owner.job_id!r}); stale ownership requires explicit "
                        "recovery"
                    ) from error
                stat = lock_path.stat()
                return cls(
                    path=lock_path,
                    owner=owner,
                    _device=stat.st_dev,
                    _inode=stat.st_ino,
                )
        finally:
            try:
                candidate_path.unlink()
            except FileNotFoundError:
                pass

    def release(self) -> None:
        """Remove only the unchanged lock inode owned by this acquisition."""

        with _directory_mutex(self.path.parent):
            if self._released:
                raise OutputLockError("this process no longer owns the output lock")
            try:
                stat = self.path.stat()
            except FileNotFoundError as error:
                raise OutputLockError(
                    "this process no longer owns the output lock"
                ) from error
            if (stat.st_dev, stat.st_ino) != (self._device, self._inode):
                raise OutputLockError(
                    "this process no longer owns the output lock inode"
                )
            current = _read_lock_owner(self.path)
            if current.fingerprint() != self.owner.fingerprint():
                raise OutputLockError(
                    "this process no longer owns the output lock record"
                )
            self.path.unlink()
            self._released = True


def _default_owner_is_alive(owner: LockOwner) -> bool | None:
    """Probe a same-host PID and refuse to guess about a remote host."""

    if owner.hostname != socket.gethostname():
        return None
    try:
        os.kill(owner.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recover_output_lock(
    output_directory: str | Path,
    *,
    expected_fingerprint: str,
    owner_is_alive: Callable[[LockOwner], bool | None] = _default_owner_is_alive,
) -> LockOwner:
    """Explicitly remove one proven-stale, fingerprint-matched ownership record."""

    directory = Path(output_directory)
    path = directory / OUTPUT_LOCK_NAME
    with _directory_mutex(directory):
        owner = _read_lock_owner(path)
        if owner.fingerprint() != expected_fingerprint:
            raise OutputLockError(
                "output lock owner fingerprint changed; refusing recovery"
            )
        alive = owner_is_alive(owner)
        if alive is True:
            raise OutputLockError(
                f"output lock owner pid {owner.pid} is still alive on {owner.hostname}"
            )
        if alive is None:
            raise OutputLockError(
                "output lock owner liveness cannot be proven from this host"
            )
        stat = path.stat()
        current = _read_lock_owner(path)
        current_stat = path.stat()
        if (
            current.fingerprint() != expected_fingerprint
            or (current_stat.st_dev, current_stat.st_ino)
            != (stat.st_dev, stat.st_ino)
        ):
            raise OutputLockError(
                "output lock changed during recovery; refusing removal"
            )
        path.unlink()
        return owner


PREEMPTION_EXIT_CODE = 75


class PreemptionFlag:
    """Process-local signal state whose handler performs no external work."""

    def __init__(self) -> None:
        self._requested = False
        self._signal_number: int | None = None

    @property
    def requested(self) -> bool:
        """Return whether this process received a preemption signal."""

        return self._requested

    @property
    def signal_number(self) -> int | None:
        """Return the most recently received signal number."""

        return self._signal_number

    def request(self, signal_number: int | None = None) -> None:
        """Set only primitive process-local state."""

        self._requested = True
        self._signal_number = signal_number

    def signal_handler(self, signal_number: int, _frame: object) -> None:
        """Record signal receipt without I/O, allocation, or collectives."""

        self._requested = True
        self._signal_number = signal_number


@dataclass
class PreemptionHandlerRegistration:
    """Prior signal handlers that must be restored after the training call."""

    previous: tuple[tuple[int, object], ...]
    _restored: bool = False

    def restore(self) -> None:
        """Restore each prior handler exactly once."""

        if self._restored:
            return
        for signal_number, handler in self.previous:
            signal.signal(signal_number, handler)  # type: ignore[arg-type]
        self._restored = True


def install_preemption_handlers(flag: PreemptionFlag) -> PreemptionHandlerRegistration:
    """Install flag-only handlers for graceful SIGUSR1 and best-effort SIGTERM."""

    previous = tuple(
        (signal_number, signal.signal(signal_number, flag.signal_handler))
        for signal_number in (signal.SIGUSR1, signal.SIGTERM)
    )
    return PreemptionHandlerRegistration(previous)


def coordinated_preemption_requested(
    flag: PreemptionFlag,
    *,
    collective_device: torch.device,
    group: dist.ProcessGroup | None = None,
) -> bool:
    """Reduce rank-local requests only when called at a completed-update safe point."""

    if not (dist.is_available() and dist.is_initialized()):
        return flag.requested
    request = torch.tensor(
        int(flag.requested),
        dtype=torch.int32,
        device=collective_device,
    )
    dist.all_reduce(request, op=dist.ReduceOp.MAX, group=group)
    return bool(request.item())


class TrainingPreempted(RuntimeError):
    """Signal a valid checkpoint boundary to the CLI with a requeue-friendly exit."""

    exit_code = PREEMPTION_EXIT_CODE

    def __init__(self, checkpoint_path: str | Path, *, update: int) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.update = update
        super().__init__(
            f"preemption checkpoint saved at update {update}: {self.checkpoint_path}"
        )
