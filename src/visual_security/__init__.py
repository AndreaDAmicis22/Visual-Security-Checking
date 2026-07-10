"""
Visual Security — PPE detection and tracking for construction sites.

Pipeline: detector open-vocabulary Apache 2.0 (Grounding DINO / OmDet-Turbo)
+ zone vietate. Nessuna dipendenza Ultralytics/YOLO (AGPL).
"""

from .analyzer import (
    AnalysisResult,
    BaseAnalyzer,
    Detection,
    GroundingDinoAnalyzer,
    OmDetTurboAnalyzer,
    build_detector,
)
from .person_ppe_checker import PersonPPEChecker, PersonPPEResult
from .person_tracker import PersonTracker
from .video_tracker import VideoSafetyTracker, build_tracker
from .zone_monitor import RestrictedZone, ZoneMonitor, ZoneViolation

__all__ = [
    "AnalysisResult",
    "BaseAnalyzer",
    "Detection",
    "GroundingDinoAnalyzer",
    "OmDetTurboAnalyzer",
    "PersonPPEChecker",
    "PersonPPEResult",
    "PersonTracker",
    "RestrictedZone",
    "VideoSafetyTracker",
    "ZoneMonitor",
    "ZoneViolation",
    "build_detector",
    "build_tracker",
]
