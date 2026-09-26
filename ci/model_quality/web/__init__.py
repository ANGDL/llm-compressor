"""Web observer and control plane for model quality runs.

The web layer reads the JSON facts written by the existing CI commands and
never keeps a second copy of run state. It executes no shell command inside a
request: write endpoints validate, authorize, audit, and record jobs for a
registered executor adapter (the local ``worker`` module or Buildkite). A
small WSGI adapter is included so the server needs no third-party framework;
deployments may wrap :class:`RunStore` with a different HTTP framework later.
"""

from .app import ModelQualityAPI, create_app, serve
from .audit import AuditLog, ConflictError, IdempotencyStore, ValidationError
from .backups import BackupService
from .control import ControlPlane
from .executors import BuildkiteExecutor, DryRunExecutor, QueueExecutor
from .jobs import JobStore, ReservationLedger
from .ops import (
    NotificationService,
    ScheduleStore,
    TrendService,
    capabilities,
    capacity_forecast,
    next_run,
)
from .plans import (
    LaunchPolicy,
    PlanRegistry,
    PlanService,
    build_quantization_argv,
    default_run_id,
    import_quantization_command,
    validate_inference_config,
)
from .publishing import PublishService
from .security import (
    AuthenticationError,
    PermissionDenied,
    Principal,
    parse_roles,
)
from .store import NotFoundError, RunStore

__all__ = [
    "AuditLog",
    "AuthenticationError",
    "BackupService",
    "BuildkiteExecutor",
    "ConflictError",
    "ControlPlane",
    "DryRunExecutor",
    "IdempotencyStore",
    "JobStore",
    "LaunchPolicy",
    "ModelQualityAPI",
    "NotFoundError",
    "NotificationService",
    "PermissionDenied",
    "PlanRegistry",
    "PlanService",
    "Principal",
    "PublishService",
    "QueueExecutor",
    "ReservationLedger",
    "RunStore",
    "ScheduleStore",
    "TrendService",
    "ValidationError",
    "build_quantization_argv",
    "capacity_forecast",
    "capabilities",
    "create_app",
    "default_run_id",
    "import_quantization_command",
    "next_run",
    "parse_roles",
    "serve",
    "validate_inference_config",
]
