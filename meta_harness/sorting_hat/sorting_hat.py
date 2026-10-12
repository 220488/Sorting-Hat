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


Includes a fallback in case of "NA" assignments (mini-swe), as well as an error fallback according to 
a fallback map.

"""

from pathlib import Path
from harbor.agents.base import BaseAgent
from sorting_hat.routing import load_route, call_route
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
import json
from harbor.agents.factory import AgentFactory
from datetime import datetime, timezone



ROUTING_STRATEGIES = {
    "rules_based": "routing_strategy.rules_based:route",
    "llm_classifier": "routing_strategy.llm_classifier:route",
    "embedding": "routing_strategy.embedding:create_router",
}

FALLBACK = "mini-swe-agent"    # harness for "NA" routes

RETRY_HARNESS_MAP = {              # first attempt -> retry
    "mini-swe-agent": "terminus-2",
    "terminus-2": "mini-swe-agent",
    "pi": "mini-swe-agent",
}

class SortingHat(BaseAgent):
    
    @staticmethod
    def name() -> str:
        return "sorting-hat"

    def version(self) -> str:
        return "0.0.1"  

    def __init__(self, 
                 logs_dir: Path, 
                 routing_strategy = "rules_based",
                 model_name: str | None = None, 
                 harnesses=("mini-swe-agent", "terminus-2", "pi"),
                 force_harness: str | None = None,
                 fallback=FALLBACK, # single best as fallback
                 retry_harness=RETRY_HARNESS_MAP,
                 harness_kwargs = None,
                 routing_kwargs = None,
                 **harbor_kwargs
                 ):
        super().__init__(logs_dir=logs_dir,
                         model_name=model_name,
                         **harbor_kwargs)
        self.routing_strategy = routing_strategy
        self.route_func = load_route(routing_strategy, ROUTING_STRATEGIES, routing_kwargs)
        self.harnesses = set(harnesses)
        self.force_harness = force_harness
        self.fallback= fallback
        self.retry_harness = retry_harness
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

    @staticmethod
    def _add_tokens(total: AgentContext, extra: AgentContext) -> None:
        for field in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd"):
            value = getattr(extra, field)
            if value is not None:
                setattr(total, field, (getattr(total, field) or 0) + value)


    async def _install(self, harness_name, environment) -> dict:
        # Install the harness, then return its install timing and version
        self.harness = AgentFactory.create_agent_from_name(
            name = harness_name,
            logs_dir = self.logs_dir,
            model_name = self.model_name,
            **self.harbor_kwargs,
            **self.harness_kwargs.get(harness_name, {})
            )
        self.harness.session_id = self.session_id
        self.harness.context_id = self.context_id

        started_at = datetime.now(timezone.utc).isoformat()
        await self.harness.setup(environment)
        # Return install timing and version
        return {"harness_install": {"started_at": started_at,
                                    "finished_at": datetime.now(timezone.utc).isoformat()},
                "harness_version": self.harness.version()}

    async def run(self,
                  instruction: str,
                  environment: BaseEnvironment,
                  context: AgentContext
                  ) -> None:

        self.run_record = {"routing_strategy": self.routing_strategy}
        
        # If force harness is defined, use that
        if self.force_harness:
            assigned_harness, matched_rule = self.force_harness, "forced"
            routing_details = {"reason": "forced"}
        
        # Otherwise call routing strategy, assign harness
        else:
            assigned_harness, matched_rule, routing_details = await call_route(self.route_func, instruction)

        # Assign fallback NA selected
        fallback_required = assigned_harness not in self.harnesses
        if fallback_required:
            assigned_harness = self.fallback

        # Log selected harness
        self.run_record.update({"assigned_harness": assigned_harness,
                                "routing_details": routing_details,
                                "matched_rule": matched_rule,
                                "fallback_used": fallback_required})
        self._write_to_log(self.run_record)

        # Install and invoke harness
        try:
            install_record = await self._install(assigned_harness, environment)
            self.run_record.update(install_record)
            self._write_to_log(self.run_record)
            await self.harness.run(instruction, environment, context)

        # Harness error: retry once with the retry harness
        except Exception as e:

            # Keep the first attempt's tokens before the retry replaces self.harness
            await environment.prepare_logs_for_host()
            if self.harness is not None:
                self.harness.populate_context_post_run(context)

            # Assign retry harness
            retry_harness = self.retry_harness[assigned_harness]

            # Log error and retry harness
            self.run_record["error"] = repr(e)
            self.run_record["retry"] = {"assigned_harness": retry_harness}
            self._write_to_log(self.run_record)

            # Install and invoke retry harness
            self.harness = None
            fresh_context = AgentContext()
            
            try:
                install_record = await self._install(retry_harness, environment)
                self.run_record["retry"].update(install_record)
                self._write_to_log(self.run_record)
                await self.harness.run(instruction, environment, fresh_context)

            # Add the retry's tokens, even if it errors
            finally:
                await environment.prepare_logs_for_host()
                if self.harness is not None:
                    self.harness.populate_context_post_run(fresh_context)
                self._add_tokens(context, fresh_context)

    def populate_context_post_run(self,
                                  context: AgentContext
                                  ) -> None:
        if self.harness is not None:
            self.harness.populate_context_post_run(context)


    
