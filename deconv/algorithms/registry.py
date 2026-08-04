from __future__ import annotations

from .wiener import WienerDeconvolution, TorchBatchWienerDeconvolution
from .richardson_lucy import (
    RichardsonLucyDeconvolution, RichardsonLucyWienerDeconvolution,
    RichardsonLucyRosenDeconvolution, TorchBatchRichardsonLucyDeconvolution,
    TorchBatchRichardsonLucyWienerDeconvolution, TorchBatchRichardsonLucyRosenDeconvolution,
)
from .landweber import (
    LandweberDeconvolution, LandweberWienerPreconditionedDeconvolution,
    TorchBatchLandweberDeconvolution,
)
from .blind import BlindRichardsonLucyDeconvolution, TorchBlindAdamTVMAPDeconvolution
from .adam import TorchAdamTVMAPDeconvolution
from .kaczmarz import BlockKaczmarzDeconvolution

class AlgorithmRegistry:
    def __init__(self) -> None:
        self.algorithms = {
            WienerDeconvolution.name: WienerDeconvolution(),
            RichardsonLucyDeconvolution.name: RichardsonLucyDeconvolution(),
            RichardsonLucyWienerDeconvolution.name: RichardsonLucyWienerDeconvolution(),
            BlindRichardsonLucyDeconvolution.name: BlindRichardsonLucyDeconvolution(),
            LandweberDeconvolution.name: LandweberDeconvolution(),
            LandweberWienerPreconditionedDeconvolution.name: LandweberWienerPreconditionedDeconvolution(),
            BlockKaczmarzDeconvolution.name: BlockKaczmarzDeconvolution(),
            RichardsonLucyRosenDeconvolution.name: RichardsonLucyRosenDeconvolution(),
            TorchBatchWienerDeconvolution.name: TorchBatchWienerDeconvolution(),
            TorchBatchRichardsonLucyDeconvolution.name: TorchBatchRichardsonLucyDeconvolution(),
            TorchBatchRichardsonLucyWienerDeconvolution.name: TorchBatchRichardsonLucyWienerDeconvolution(),
            TorchBatchRichardsonLucyRosenDeconvolution.name: TorchBatchRichardsonLucyRosenDeconvolution(),
            TorchBatchLandweberDeconvolution.name: TorchBatchLandweberDeconvolution(),
            TorchAdamTVMAPDeconvolution.name: TorchAdamTVMAPDeconvolution(),
            TorchBlindAdamTVMAPDeconvolution.name: TorchBlindAdamTVMAPDeconvolution(),
        }

    def names(self) -> List[str]:
        return list(self.algorithms.keys())

    def get(self, name: str) -> DeconvolutionAlgorithm:
        return self.algorithms[name]


__all__ = ["AlgorithmRegistry"]
