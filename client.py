from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .src.models import BugTriageAction, BugTriageObservation

class BugTriageEnv(EnvClient[BugTriageAction, BugTriageObservation, State]):
    """
    Client for the Bug Triage Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    """

    def _step_payload(self, action: BugTriageAction) -> Dict:
        """Convert action to JSON payload."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[BugTriageObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = BugTriageObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
