"""Physics stopgap: codar-sounder's Kaeppler pipeline (virtual height, layer,
codar.spots), carried here until the central side produces the same
quantities.  The modules move unchanged apart from import paths; codar-sounder's
own copy retires with codar-sounder (Michael, 2026-11-19, decision 2)."""
from chirp_recorder.physics.stopgap import PhysicsOut, PhysicsStopgap

__all__ = ["PhysicsOut", "PhysicsStopgap"]
