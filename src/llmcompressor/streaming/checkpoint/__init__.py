"""On-demand access to tensors in safetensors checkpoints."""

from .coordinator import (
    CommittedTransactionCorruptionError,
    SingleParticipantCoordinator,
    StreamingWorkDirectoryLock,
    TransactionCoordinationError,
    fault_point,
    with_work_directory_lock,
)
from .weight_map import TensorMetadata, WeightMap
from .weight_source import CheckpointWeightSource, SafetensorsWeightSource
from .writer import (
    AsyncDirectSafetensorsWriter,
    DirectSafetensorsWriter,
    StreamingCheckpointWriter,
    TensorRecord,
)

__all__ = [
    "CheckpointWeightSource",
    "CommittedTransactionCorruptionError",
    "AsyncDirectSafetensorsWriter",
    "DirectSafetensorsWriter",
    "SafetensorsWeightSource",
    "StreamingCheckpointWriter",
    "SingleParticipantCoordinator",
    "StreamingWorkDirectoryLock",
    "TransactionCoordinationError",
    "TensorMetadata",
    "TensorRecord",
    "WeightMap",
    "fault_point",
    "with_work_directory_lock",
]
