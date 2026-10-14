from __future__ import annotations

from pathlib import Path

import yaml

from tdex.boundary import ClosureConfig
from tdex.render import RenderConfig

REPO = Path(__file__).resolve().parents[1]

GR1_TASKS = [  # the 24 tasks in RLDX-1 run_scripts/eval/gr1_tabletop/eval_gr1.sh, same order
    "PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
]


def load(path: str | Path = REPO / "configs/gr1.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def closure_config(cfg: dict) -> ClosureConfig:
    c = cfg["closure"]
    return ClosureConfig(**{**c, "flexion_dims": tuple(c["flexion_dims"])})


def render_config(cfg: dict) -> RenderConfig:
    return RenderConfig(**cfg.get("render", {}))
