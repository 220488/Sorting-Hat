"""Based on Harbor documentation:

1) Harbor Custom Agents, External Agents: https://docs.harborframework.com/agents/custom-agents#external-agent

2) Harbor Custom Agents Example (Marker Agent): https://github.com/harbor-framework/harbor/blob/v0.22.0/examples/agents/marker_agent.py#L19-L66

3) Harbor AgentFactory: https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/agents/factory.py


Running the meta-router with Harbor on Terminal Bench 2.1:

PYTHONPATH=. harbor run \
  -d terminal-bench/terminal-bench-2-1@<dataset version> \
  -a sorting_hat.sorting_hat:SortingHat \
  -m <baseline model string> \
  --ak routing_strategy=rules_based \
  --ak 'harness_kwargs={"mini-swe-agent": {"version": "<v>"}, "pi": {"version": "<v>"}}' \
  -n <baseline concurrency>

Includes a fallback in case of "NA" assignments (mini-swe).

"""

from pathlib import Path
from harbor.agents.base import BaseAgent
import importlib
import inspect
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
import json
from harbor.agents.factory import AgentFactory
from datetime import datetime, timezone



ROUTING_STRATEGIES = {
    "rules_based": "routing_strategy.rules_based:route",
    "llm_classifier": "routing_strategy.llm_classifier:route",
}

class SortingHatNoRetry(BaseAgent):
    
    @staticmethod
    def name() -> str:
        return "sorting-hat-no-retry"

    def version(self) -> str:
        return "0.0.1"  

    def __init__(self, 
                 logs_dir: Path, 
                 routing_strategy = "rules_based",
                 model_name: str | None = None, 
                 harnesses=("mini-swe-agent", "terminus-2", "pi"),
                 force_harness: str | None = None,
                 fallback="mini-swe-agent", # single best as fallback
                 harness_kwargs = None,
                 **harbor_kwargs
                 ):
        super().__init__(logs_dir=logs_dir,
                         model_name=model_name,
                         **harbor_kwargs)
        self.routing_strategy = routing_strategy
        routing_strategy_module_path, routing_func = ROUTING_STRATEGIES[routing_strategy].split(":")
        self.route_func = getattr(importlib.import_module(routing_strategy_module_path), routing_func)
        self.harnesses = set(harnesses)
        self.force_harness = force_harness
        self.fallback= fallback
        self.harness = None
        self.harness_kwargs = harness_kwargs or {}
        self.harbor_kwargs = harbor_kwargs
        self.sorting_hat_log = "sorting_hat_log.json"

    async def setup(self,
                    environment: BaseEnvironment
                    ) -> None:
        pass

    def _write_to_log(self, record: dict) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)  
        (self.logs_dir / self.sorting_hat_log).write_text(json.dumps(record, indent=2, default=str))  

    
    async def run(self,
                  instruction: str,
                  environment: BaseEnvironment,
                  context: AgentContext
                  ) -> None:

        # If force harness is defined, use that
        if self.force_harness:
            assigned_harness, matched_rule = self.force_harness, "forced"
        
        # Otherwise call routing strategy, assign harness
        else:
            try:
                route_result = self.route_func(instruction)
                if inspect.isawaitable(route_result):
                    route_result = await route_result
                assigned_harness, matched_rule = route_result
            except Exception:
                assigned_harness, matched_rule = "NA", "NA"

        # Assign fallback if needed
        fallback_required = assigned_harness not in self.harnesses
        if fallback_required:
            assigned_harness = self.fallback

        # Log selected harness
        run_record = {"routing_strategy": self.routing_strategy,
                    "assigned_harness": assigned_harness,
                    "matched_rule": matched_rule,
                    "fallback_used": fallback_required,
                    }
        self._write_to_log(run_record)

        # Install the assigned harness
        self.harness = AgentFactory.create_agent_from_name(
            name = assigned_harness,
            logs_dir = self.logs_dir,
            model_name = self.model_name,
            **self.harbor_kwargs,
            **self.harness_kwargs.get(assigned_harness, {})
            )
        self.harness.session_id = self.session_id
        self.harness.context_id = self.context_id

        started_at = datetime.now(timezone.utc).isoformat()
        await self.harness.setup(environment)
        run_record["harness_install"] = {"started_at": started_at,
                                         "finished_at": datetime.now(timezone.utc).isoformat()
                                         }
        run_record["harness_version"] = self.harness.version()
        self._write_to_log(run_record)

        # Invoke harness
        await self.harness.run(instruction, environment, context)

    def populate_context_post_run(self,
                                  context: AgentContext
                                  ) -> None:
        if self.harness is not None:
            self.harness.populate_context_post_run(context)


    
