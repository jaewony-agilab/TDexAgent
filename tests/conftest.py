import numpy as np
import pytest

from tdex.boundary import ClosureConfig
from tdex.types import Boundary, BoundaryPacket, FingerSummary


@pytest.fixture
def closure():
    return ClosureConfig(flexion_dims=(1, 2, 3, 4), open_value=0.0, closed_value=1.0,
                         settle_vel_eps=1e-3, settle_steps=2, max_settle_steps=10)


def summaries(hand="right", contact=("thumb", "index")):
    return [
        FingerSummary(hand, f, (0.0, 0.0, 2.0 if f in contact else 0.0), 2.0 if f in contact else 0.0,
                      3.0 if f in contact else 0.0, f in contact, False)
        for f in ("thumb", "index", "middle", "ring", "pinky", "palm")
    ]


@pytest.fixture
def packet():
    img = np.zeros((32, 32, 3), np.uint8)
    return BoundaryPacket(
        boundary=Boundary("right", "close", 10),
        frames={"scene": img, "wrist": img},
        overlay_frames={"scene": img + 1, "wrist": img + 1},
        finger_summaries=summaries(),
        task_instruction="pick the can and place it in the drawer",
        retry_count=0,
        meta={"task": "T", "episode": 0},
    )
