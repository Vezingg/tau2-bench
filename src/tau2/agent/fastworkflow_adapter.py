# FastWorkflow Agent Adapter for Tau2 Bench
#
# Architecture (Option A – propose-only with tool replay):
#
# FastWorkflow runs its internal DSPy ReAct agent in a background thread,
# executing tools via CommandExecutor.invoke_command() synchronously.  Trace
# events (WORKFLOW_TO_AGENT) record each tool's name, arguments and result.
#
# This adapter:
#   1. Blocks on command_output_queue until FastWorkflow finishes processing.
#   2. Collects all WORKFLOW_TO_AGENT trace events emitted during processing.
#   3. Filters traces to Tau2-known tools and returns them as ToolCall objects
#      so Tau2's environment executes them (keeping env state in sync for the
#      evaluator).
#   4. After Tau2's environment returns ToolMessages, the adapter returns the
#      final text response from FastWorkflow.
#   5. If FastWorkflow calls ask_user() mid-processing, the adapter detects
#      the intermediate output, replays any tools collected so far, and then
#      returns the clarification text to Tau2's user simulator.  The next
#      UserMessage is forwarded to FastWorkflow's user_message_queue so
#      processing can continue.

import contextlib
import json
import logging
import os
import time
import queue
import copy
from typing import List, Dict, Any, Optional, Tuple, Set
from dotenv import dotenv_values

from tau2.agent.base import BaseAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import get_cost

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter phases
# ---------------------------------------------------------------------------
_PHASE_IDLE = "idle"                    # ready for new UserMessage
_PHASE_REPLAYING_TOOLS = "replaying"    # returned tool calls, awaiting results
_PHASE_RETURNING_TEXT = "returning"     # all tools replayed, return final text


def _json_deepcopy(obj: Any) -> Any:
    """Safe deep copy using JSON serialization with fallback."""
    with contextlib.suppress(Exception):
        return json.loads(json.dumps(obj))
    return copy.deepcopy(obj)


class FastWorkflowAgentAdapter(BaseAgent):
    """
    FastWorkflow agent adapter that integrates with Tau2 Bench.

    This adapter bridges FastWorkflow's command-trace queue architecture with
    Tau2 Bench's message-based orchestration system using a *propose-only*
    strategy:

    * FastWorkflow's internal ReAct agent executes tools against its own
      state (via ``CommandExecutor``).
    * The adapter collects the WORKFLOW_TO_AGENT trace events and re-emits
      them as Tau2 ``ToolCall`` objects so Tau2's environment executes the
      same tools, keeping its state in sync for evaluation.
    * The adapter uses a phase-based state machine
      (``idle`` → ``replaying`` → ``idle``) to interleave tool-call replay
      with final text responses.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        model: str = "mistral-small-latest",
        provider: str = "mistral",
        temperature: float = 0.0,
        use_reasoning: bool = True,
        workflow_type: str = "retail",
        fw_output_timeout: float = 300.0,
        **kwargs,
    ):
        """
        Initialize the FastWorkflow adapter.

        Args:
            tools: List of Tau2 Tool objects available to the agent.
            domain_policy: Domain-specific policy text.
            model: LLM model name (default: mistral-small-latest).
            provider: LLM provider (default: mistral).
            temperature: Model temperature (default: 0.0).
            use_reasoning: Enable reasoning mode (default: True).
            workflow_type: Type of workflow to use (retail/airline/telecom).
            fw_output_timeout: Max seconds to wait for FastWorkflow output
                per turn (default: 300 – generous for multi-step LLM calls).
            **kwargs: Additional configuration.
        """
        self.tools = tools
        self.domain_policy = domain_policy
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.workflow_type = workflow_type
        self.fw_output_timeout = fw_output_timeout

        # Build a set of Tau2-known tool names so we can filter traces
        self._tau2_tool_names: Set[str] = {t.name for t in tools}

        # Find the workflow path
        self.workflow_path = self._find_workflow_path(workflow_type)

        # FastWorkflow session (initialized per task)
        self.fastworkflow = None
        self.chat_session = None
        self.is_initialized = False

        logger.info(f"FastWorkflow adapter initialized for {workflow_type} domain")
        logger.info(f"Model: {model} from {provider}")
        logger.info(f"Workflow path: {self.workflow_path}")
        logger.info(f"Tau2 tools ({len(self._tau2_tool_names)}): {sorted(self._tau2_tool_names)}")

    # ------------------------------------------------------------------
    # Workflow helpers
    # ------------------------------------------------------------------

    def _find_workflow_path(self, workflow_type: str) -> str:
        """Find the path to the specified workflow."""
        current_dir = os.getcwd()
        workflow_path = os.path.join(current_dir, "examples", f"{workflow_type}_workflow")

        if os.path.exists(workflow_path):
            return workflow_path

        raise FileNotFoundError(
            f"Could not find {workflow_type} workflow. Expected at: {workflow_path}. "
            f"Run 'fastworkflow examples fetch {workflow_type}_workflow' to install it."
        )

    def _initialize_fastworkflow(self, initial_message: Optional[str] = None):
        """Initialize FastWorkflow session if not already initialized."""
        if self.is_initialized:
            return

        try:
            # Load environment variables
            env_vars = {
                **dotenv_values('examples/fastworkflow.env'),
                **dotenv_values('examples/fastworkflow.passwords.env'),
            }

            # Import and initialize FastWorkflow
            import fastworkflow
            self.fastworkflow = fastworkflow
            fastworkflow.init(env_vars=env_vars)
            logger.info("✅ FastWorkflow initialized")

            # Clear any lingering workflow stack
            with contextlib.suppress(Exception):
                fastworkflow.ChatSession.clear_workflow_stack()

            # Create chat session (agent mode)
            self.chat_session = fastworkflow.ChatSession(run_as_agent=True)
            logger.info("✅ Chat session created")

            # Start workflow – keep_alive=True so the background thread persists
            if initial_message:
                self.chat_session.start_workflow(
                    self.workflow_path,
                    workflow_context=None,
                    startup_command=initial_message,
                    startup_action=None,
                    keep_alive=True,
                    project_folderpath=None,
                )
                logger.info(f"✅ Workflow started with message: {initial_message}")

            self.is_initialized = True

        except Exception as e:
            logger.error(f"❌ Error initializing FastWorkflow: {e}")
            raise

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_plain_kwargs(params: Any) -> Dict[str, Any]:
        """Convert parameters to plain dict."""
        if params is None:
            return {}
        if isinstance(params, dict):
            return params
        with contextlib.suppress(Exception):
            return params.model_dump()
        with contextlib.suppress(Exception):
            return params.dict()
        with contextlib.suppress(Exception):
            return dict(params)
        return {}

    def _drain_command_trace(
        self,
        max_drain: int = 500,
    ) -> List[Tuple[str, Dict[str, Any], str, bool]]:
        """
        Drain the command_trace_queue and return WORKFLOW_TO_AGENT commands.

        Returns:
            List of tuples ``(command_name, parameters, response_text, success)``
        """
        if not self.is_initialized or not self.chat_session:
            return []

        executed_commands: list[tuple[str, dict, str, bool]] = []
        processed = 0

        while processed < max_drain:
            try:
                evt = self.chat_session.command_trace_queue.get_nowait()
            except queue.Empty:
                break

            processed += 1

            # Skip AGENT_TO_WORKFLOW direction (these are the raw command
            # strings sent *to* the workflow, not the executed results).
            is_agent_to_workflow = (
                getattr(evt, "direction", None)
                == getattr(self.fastworkflow, "CommandTraceEventDirection", type(None)).AGENT_TO_WORKFLOW
                if hasattr(self.fastworkflow, "CommandTraceEventDirection")
                else False
            )

            if not is_agent_to_workflow:
                cmd_name = getattr(evt, "command_name", None)
                params = self._to_plain_kwargs(getattr(evt, "parameters", None))
                response_text = getattr(evt, "response_text", None)
                success = getattr(evt, "success", True)

                if isinstance(cmd_name, str) and cmd_name:
                    executed_commands.append((cmd_name, params, response_text or "", success))

        return executed_commands

    def _push_user_message(self, user_text: str):
        """Push user message to FastWorkflow's user_message_queue."""
        if self.is_initialized and self.chat_session:
            self.chat_session.user_message_queue.put(user_text)
            logger.debug(f"Pushed user message to FastWorkflow: {user_text[:100]}...")

    # ------------------------------------------------------------------
    # Blocking output collection
    # ------------------------------------------------------------------

    def _wait_for_fw_output(
        self,
        timeout: Optional[float] = None,
    ) -> Tuple[Optional[str], List[Tuple[str, Dict[str, Any], str, bool]]]:
        """
        Block until FastWorkflow produces output, collecting traces meanwhile.

        This polls ``command_output_queue`` with a short timeout in a loop,
        draining ``command_trace_queue`` between each attempt so no trace
        events are lost.

        Args:
            timeout: Max seconds to wait (defaults to ``self.fw_output_timeout``).

        Returns:
            ``(output_text, traces)`` – the text from the ``CommandOutput``
            and the list of WORKFLOW_TO_AGENT traces collected during processing.
        """
        if timeout is None:
            timeout = self.fw_output_timeout

        all_traces: list[tuple[str, dict, str, bool]] = []
        start = time.monotonic()
        output_text: Optional[str] = None

        while time.monotonic() - start < timeout:
            # Drain traces accumulated so far
            new_traces = self._drain_command_trace()
            all_traces.extend(new_traces)

            # Try to get an output from FastWorkflow
            try:
                out = self.chat_session.command_output_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Extract text from CommandOutput
            if hasattr(out, "command_responses") and isinstance(out.command_responses, list):
                texts = [
                    cr.response.strip()
                    for cr in out.command_responses
                    if getattr(cr, "response", None) and cr.response.strip()
                ]
                output_text = "\n".join(texts) if texts else None

            # Give a brief window for any remaining traces emitted just
            # before the output was posted to the queue.
            time.sleep(0.2)
            remaining = self._drain_command_trace()
            all_traces.extend(remaining)
            break

        if output_text is None:
            logger.warning(f"⚠️ No FastWorkflow output after {timeout}s timeout")

        return output_text, all_traces

    # ------------------------------------------------------------------
    # Trace → ToolCall conversion
    # ------------------------------------------------------------------

    def _traces_to_tool_calls(
        self,
        traces: List[Tuple[str, Dict[str, Any], str, bool]],
    ) -> List[ToolCall]:
        """
        Convert WORKFLOW_TO_AGENT trace events to Tau2 ``ToolCall`` objects.

        Only traces whose ``command_name`` matches a Tau2 environment tool are
        included; internal FastWorkflow commands (``abort``, ``what_can_i_do``,
        etc.) are silently skipped.
        """
        tool_calls: list[ToolCall] = []
        for cmd_name, params, _response_text, _success in traces:
            if cmd_name not in self._tau2_tool_names:
                logger.debug(
                    f"Skipping non-Tau2 trace command: {cmd_name}"
                )
                continue
            tc = ToolCall(
                id=f"call_{len(tool_calls)}",
                name=cmd_name,
                arguments=params or {},
                requestor="assistant",
            )
            tool_calls.append(tc)
        return tool_calls

    # ------------------------------------------------------------------
    # Core interface – generate_next_message
    # ------------------------------------------------------------------

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: Any,
    ) -> Tuple[AssistantMessage, Any]:
        """
        Generate the next message from FastWorkflow agent.

        Phase-based state machine:

        ``idle``
            Ready for a new ``UserMessage``.  The adapter sends the message
            to FastWorkflow, blocks until output is produced, collects
            traces, and returns either tool calls (→ ``replaying``) or
            text (stays ``idle``).

        ``replaying``
            We previously returned tool calls so Tau2's environment could
            execute them.  Now we receive ``ToolMessage``/``MultiToolMessage``
            with the results (which we discard – FastWorkflow already has
            its own results).  We return the pending text response and
            transition back to ``idle``.

        Args:
            message: ``UserMessage``, ``ToolMessage``, or ``MultiToolMessage``.
            state: Agent state dict (maintained across calls).

        Returns:
            ``(AssistantMessage, updated_state)``
        """
        try:
            # ----- bootstrap state -----
            if state is None:
                state = self._make_fresh_state()

            phase = state.get("phase", _PHASE_IDLE)

            # =============================================================
            # PHASE: replaying_tools
            #   We previously returned ToolCalls.  Tau2's environment has
            #   executed them and is sending back ToolMessages.  We ignore
            #   the results and return the pending text.
            # =============================================================
            if phase == _PHASE_REPLAYING_TOOLS:
                state["message_history"].append(message)
                text = state.pop("pending_text", None) or "Task completed."
                state["phase"] = _PHASE_IDLE
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content=text,
                    tool_calls=None,
                    cost=0.0,
                )
                state["message_history"].append(assistant_msg)
                logger.info(f"🤖 Returning text after tool replay: {text[:120]}...")
                return assistant_msg, state

            # =============================================================
            # PHASE: idle – process new input
            # =============================================================

            # --- UserMessage (initial or follow-up) ---
            if isinstance(message, UserMessage):
                state["message_history"].append(message)
                user_content = message.content

                if not self.is_initialized:
                    logger.info(f"🎯 Starting FastWorkflow with: {user_content}")
                    self._initialize_fastworkflow(initial_message=user_content)
                else:
                    logger.info(f"👤 User says: {user_content}")
                    self._push_user_message(user_content)

                # Block until FastWorkflow produces output
                output_text, traces = self._wait_for_fw_output()

                # Convert traces to Tau2 ToolCalls (filtered to known tools)
                tool_calls = self._traces_to_tool_calls(traces)

                if tool_calls:
                    # Return tool calls first; store text for next round
                    state["pending_text"] = output_text or "Task completed."
                    state["phase"] = _PHASE_REPLAYING_TOOLS
                    assistant_msg = AssistantMessage(
                        role="assistant",
                        content=None,
                        tool_calls=tool_calls,
                        cost=0.0,
                    )
                    logger.info(
                        f"🔧 Returning {len(tool_calls)} tool calls for replay; "
                        f"text buffered for next step"
                    )
                elif output_text:
                    state["phase"] = _PHASE_IDLE
                    assistant_msg = AssistantMessage(
                        role="assistant",
                        content=output_text,
                        tool_calls=None,
                        cost=0.0,
                    )
                    logger.info(f"🤖 Agent says (no tools): {output_text[:120]}...")
                else:
                    state["phase"] = _PHASE_IDLE
                    assistant_msg = AssistantMessage(
                        role="assistant",
                        content=(
                            "I apologize, but I was unable to process your "
                            "request. Could you please rephrase?"
                        ),
                        tool_calls=None,
                        cost=0.0,
                    )
                    logger.warning("⚠️ No output or traces from FastWorkflow")

                state["message_history"].append(assistant_msg)
                return assistant_msg, state

            # --- ToolMessage / MultiToolMessage in idle phase ---
            # This can happen if the orchestrator sends a ToolMessage that
            # doesn't correspond to a replay cycle (e.g. edge cases).
            elif isinstance(message, ToolMessage) or hasattr(message, "tool_messages"):
                state["message_history"].append(message)
                logger.debug("Received ToolMessage in idle phase – acknowledging")
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content="Understood, continuing.",
                    tool_calls=None,
                    cost=0.0,
                )
                state["message_history"].append(assistant_msg)
                return assistant_msg, state

            # --- Fallback for unknown message types ---
            else:
                logger.warning(f"Unexpected message type: {type(message)}")
                state["message_history"].append(message)
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content="I'm sorry, I didn't understand that input.",
                    tool_calls=None,
                    cost=0.0,
                )
                state["message_history"].append(assistant_msg)
                return assistant_msg, state

        except Exception as e:
            logger.error(f"❌ Error in generate_next_message: {e}")
            import traceback
            traceback.print_exc()
            error_msg = AssistantMessage(
                role="assistant",
                content=f"I encountered an error: {str(e)}",
                tool_calls=None,
                cost=0.0,
            )
            return error_msg, state or self._make_fresh_state()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_fresh_state() -> Dict[str, Any]:
        return {
            "message_history": [],
            "phase": _PHASE_IDLE,
            "pending_text": None,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the FastWorkflow session for a new task.

        This ensures task isolation by stopping the workflow, clearing the
        workflow stack, and resetting the initialized flag so the next task
        starts fresh.
        """
        logger.info("🔄 Resetting FastWorkflow agent for new task")
        if self.is_initialized and self.chat_session:
            with contextlib.suppress(Exception):
                self.chat_session.stop_workflow()
            with contextlib.suppress(Exception):
                if self.fastworkflow:
                    self.fastworkflow.ChatSession.clear_workflow_stack()
            self.chat_session = None
        self.is_initialized = False
        logger.info("✅ FastWorkflow agent reset complete")

    def stop(
        self,
        message: Optional[ValidAgentInputMessage] = None,
        state: Optional[Any] = None,
    ) -> None:
        """Stop the agent and cleanup resources."""
        logger.info("🛑 Stopping FastWorkflow agent")
        self.reset()

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None,
    ) -> Any:
        """Get the initial state of the agent."""
        st = self._make_fresh_state()
        if message_history:
            st["message_history"] = list(message_history)
        return st

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """Check if the message is a stop signal."""
        if message.content and "###STOP###" in message.content:
            return True
        return False

    def set_seed(self, seed: int):
        """Set the seed for the agent."""
        logger.info(f"Setting seed {seed} for FastWorkflow adapter")
