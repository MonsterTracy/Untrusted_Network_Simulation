import json
import re
from random import Random

from werewolf.agents.llm_agent import (
    BeliefValidationError,
    GameplayActionValidationError,
    GameplayGenerationExhausted,
    GameplaySpeechQualityError,
    LLMAgent,
    RoleReportValidationError,
    belief_response_format,
    day_cognition_response_format,
    night_action_response_format,
    parse_belief_response,
    parse_day_cognition_response,
    parse_vote_response,
    validate_gameplay_public_speech,
    validate_role_report,
    vote_response_format,
)
from werewolf.agents.prompt_template_v0 import (
    DIRECT_SPEECH_TOM_PREFIX,
    DISCUSSION_TACTICS,
    DiscussionAct,
    STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE,
    build_belief_prompt,
    build_day_cognition_prompt,
    build_direct_speech_prompt,
    build_discussion_tactic_prompt,
    build_public_speech_realization_prompt,
    build_public_claim_catalog,
    build_vote_prompt,
    compile_discussion_intent,
    derive_belief_constraints,
    derive_discussion_vote_targets,
    freeze_discussion_candidates,
)
from werewolf.backends import BackendError
from werewolf.backends.openai_compatible import model_request_details
from werewolf.canonical_collection.pre import SpeakerPREBeliefHandoff
from werewolf.runtime_config import GAMEPLAY_DECISION_METHODS, MULTIMIND_METHOD
from werewolf.speech.validation import normalize_player


_CONSTRAINED_NIGHT_PHASES = (
    "skill_wolf",
    "skill_seer",
    "skill_witch",
)
GAMEPLAY_GENERATION_MAX_ATTEMPTS = 3
_GAMEPLAY_GENERATION_ERRORS = (
    BeliefValidationError,
    GameplayActionValidationError,
    GameplaySpeechQualityError,
)


class GPTAgent(LLMAgent):
    def __init__(
        self,
        backend=None,
        model_name=None,
        temperature=1.0,
        log_file=None,
        gameplay_prompt_profile=STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE,
        gameplay_max_tokens=None,
        decision_method=None,
        game_seed=None,
    ):
        if decision_method is not None and (
            not isinstance(decision_method, str) or decision_method not in GAMEPLAY_DECISION_METHODS
        ):
            raise ValueError(f"unsupported agent decision_method: {decision_method!r}")
        if decision_method == "dpins:random" and type(game_seed) is not int:
            raise ValueError("dpins:random requires an integer game seed")
        super().__init__(
            backend=backend,
            model_name=model_name,
            temperature=temperature,
            log_file=log_file,
            gameplay_prompt_profile=gameplay_prompt_profile,
            gameplay_max_tokens=gameplay_max_tokens,
        )
        self.temperature = temperature
        self.decision_method = decision_method
        self.game_seed = game_seed

    def act(self, observation):
        return self._act(observation, pre_speech_belief=None)

    def act_with_direct_tom(self, observation, *, payload, prediction):
        """同一次 ReAct 狼人发言请求可额外接收当前 PRE 的私有预测。"""
        if (self.decision_method != "react" or observation.get("identity") != "Werewolf"
                or not observation.get("phase", "").endswith(("_day_speech", "_day_speech_pk"))
                or observation.get("current_act_idx") not in observation["authoritative_public_state"]["alive_players"]):
            raise ValueError("Direct-ToM requires a living ReAct wolf's speech")
        return self._act(observation, pre_speech_belief=None, direct_tom=(payload, prediction))

    def act_with_pre_speech_belief(
        self,
        observation,
        *,
        pre_speech_belief,
        tom_context=None,
    ):
        """校验当前发言交接。默认认知分支使用冻结的 PRE 自身报告。"""

        if not isinstance(pre_speech_belief, SpeakerPREBeliefHandoff):
            raise TypeError(
                "pre_speech_belief must be SpeakerPREBeliefHandoff"
            )
        if self.gameplay_prompt_profile != STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE:
            raise ValueError(
                "PRE-belief cognition handoff requires strict_classic7 gameplay"
            )
        if normalize_player(observation.get("current_act_idx")) != (
            pre_speech_belief.observer_id
        ):
            raise ValueError("PRE-belief observer does not match current speaker")
        if "speech" not in observation.get("phase", ""):
            raise ValueError("PRE-belief handoff is only valid for speech")
        if self.decision_method is not None and tom_context is not None:
            raise ValueError("decision_method does not accept ToM context")
        if tom_context is not None and (not isinstance(tom_context, str) or not tom_context):
            raise TypeError("ToM context must be non-empty text")
        return self._act(
            observation,
            pre_speech_belief=pre_speech_belief,
            tom_context=tom_context,
        )

    def _act(self, observation, *, pre_speech_belief, tom_context=None, direct_tom=None):
        phase = observation["phase"]
        is_speech = "speech" in phase
        speech_kind = "speech_pk" if "speech_pk" in phase else "speech"
        is_vote = "vote" in phase
        is_night = any(name in phase for name in _CONSTRAINED_NIGHT_PHASES)
        if (
            self.decision_method is not None
            and observation.get("observer_id") != observation["current_act_idx"]
        ):
            raise ValueError("decision_method requires the current actor's observation")
        if (
            is_speech
            and self.decision_method is None
            and not isinstance(pre_speech_belief, SpeakerPREBeliefHandoff)
        ):
            raise TypeError(
                "strict day cognition requires SpeakerPREBeliefHandoff"
            )

        temperature, max_tokens = self._request_limits()

        if is_speech:
            if self.decision_method is not None:
                if (
                    not phase.endswith(f"_day_{speech_kind}")
                    or observation.get("valid_action") != (speech_kind, -1)
                ):
                    raise ValueError("decision_method requires a legal speech opportunity")
                if self.decision_method == MULTIMIND_METHOD:
                    raise ValueError("MultiMind speech requires the Gameplay-only Planner")
                return speech_kind, self._generate_direct_speech(
                    observation,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    direct_tom=direct_tom,
                )
            day_cognition, candidate_snapshot, claim_catalog = (
                self._generate_day_cognition(
                    observation,
                    pre_speech_belief=pre_speech_belief,
                    tom_context=tom_context,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            discussion_acts = compile_discussion_intent(
                candidate_snapshot,
                public_content_action_indices=(
                    day_cognition.public_content_action_indices
                ),
                public_vote_stance_index=(
                    day_cognition.public_vote_stance_index
                ),
            )
            raw_text = self._generate_public_speech(
                observation,
                discussion_acts=discussion_acts,
                claim_catalog=claim_catalog,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return speech_kind, raw_text

        if is_vote:
            belief = None
            if self.decision_method != "react":
                belief = self._generate_belief(
                    observation,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return self._generate_vote(
                observation,
                belief=belief,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if is_night:
            return self._generate_night_action(
                observation,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        raise ValueError(f"unsupported gameplay phase: {phase!r}")

    def realize_multimind_speech(self, observation, *, actions, belief, temperature, max_tokens):
        """按有序 V1 计划实现发言，委托现有的生成及文本质量校验。"""
        discussion_acts = tuple(
            DiscussionAct(action.action, int(action.object.removeprefix("player")))
            for action in actions
        )
        return self._generate_public_speech(
            observation, discussion_acts=discussion_acts,
            claim_catalog=build_public_claim_catalog(observation),
            temperature=temperature, max_tokens=max_tokens, multimind_belief=belief.as_dict(),
        )

    def _request_limits(self):
        is_o1 = self.model_name is not None and "o1" in self.model_name
        temperature = None if is_o1 else self.temperature
        max_tokens = self.gameplay_max_tokens
        if max_tokens is None and is_o1:
            max_tokens = 32000
        return temperature, max_tokens

    def _retry_validated_generation(self, *, stage, generate):
        last_error = None
        for attempt in range(1, GAMEPLAY_GENERATION_MAX_ATTEMPTS + 1):
            try:
                with model_request_details(kind=stage, attempt=attempt):
                    return generate(attempt, last_error)
            except _GAMEPLAY_GENERATION_ERRORS as exc:
                last_error = exc
        raise GameplayGenerationExhausted(
            stage=stage,
            attempts=GAMEPLAY_GENERATION_MAX_ATTEMPTS,
            last_error=last_error,
        ) from last_error

    def _generate_belief(self, observation, *, temperature, max_tokens):
        return self._retry_validated_generation(
            stage="belief",
            generate=lambda attempt, _last_error: self._generate_belief_once(
                observation,
                temperature=temperature,
                max_tokens=max_tokens,
                attempt=attempt,
            ),
        )

    def _generate_belief_once(
        self,
        observation,
        *,
        temperature,
        max_tokens,
        attempt,
    ):
        player_id = observation.get("current_act_idx")
        phase = observation.get("phase")
        exact_roles, role_options = derive_belief_constraints(observation)
        prompt = build_belief_prompt(
            observation,
            exact_roles=exact_roles,
            role_options=role_options,
        )
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={
                "stage": "belief",
                "observation": observation,
                "gen_times": attempt - 1,
            },
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=belief_response_format(
                supports_json_schema=getattr(self.backend, "supports_json_schema", False),
                role_options=role_options,
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if metadata["finish_reason"] == "length":
            raise BeliefValidationError(
                f"belief response was truncated (player={player_id}, phase={phase!r})"
            )
        report = parse_belief_response(
            content,
            player_id=player_id,
            self_role=observation.get("identity"),
            phase=phase,
            exact_roles=exact_roles,
            role_options=role_options,
        )
        try:
            validate_role_report(
                report,
                player_id=player_id,
                self_role=observation.get("identity"),
                phase=phase,
                exact_roles=exact_roles,
                role_options=role_options,
            )
        except RoleReportValidationError as exc:
            if self.decision_method is not None:
                raise BeliefValidationError(str(exc)) from exc
            # The raw response is already retained by the existing call audit.
            pass
        return report

    def _generate_direct_speech(self, observation, *, temperature, max_tokens, direct_tom=None):
        legal_targets = derive_discussion_vote_targets(observation)
        belief = None
        if self.decision_method != "react":
            belief = self._generate_belief(
                observation, temperature=temperature, max_tokens=max_tokens
            ).as_dict()

        tactic = None
        if self.decision_method == "dpins:llm":
            prompt = build_discussion_tactic_prompt(observation, belief)
            tactic = self._retry_validated_generation(
                stage="discussion_tactic",
                generate=lambda attempt, _last_error: self._select_discussion_tactic(
                    observation,
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    attempt=attempt,
                ),
            )
        elif self.decision_method == "dpins:random":
            # Classic7 每个座位每天至多有一次普通发言和一次 PK 发言。
            actor = observation["current_act_idx"]
            phase = observation["phase"]
            speech_seed = f"{self.game_seed}:{actor}:{phase}"
            tactic = Random(speech_seed).choice(tuple(DISCUSSION_TACTICS))

        prompt = build_direct_speech_prompt(observation, belief=belief, tactic=tactic, legal_targets=legal_targets)
        receipt = None
        if direct_tom is not None:
            payload, prediction = direct_tom
            if not isinstance(payload, str) or not payload:
                raise ValueError("Direct-ToM requires a valid prediction payload")
            prompt += DIRECT_SPEECH_TOM_PREFIX
            start = len(prompt)
            prompt += payload
            receipt = {key: prediction[key] for key in ("boundary_id", "prefix_digest", "request_id")}
            receipt["span"] = (start, len(prompt))
        with model_request_details(strategy=tactic, direct_tom=receipt):
            return self._generate_public_speech(
                observation,
                temperature=temperature,
                max_tokens=max_tokens,
                direct_prompt=prompt,
            )

    def _decision_response_format(self, name, **properties):
        if getattr(self.backend, "supports_json_schema", False) is not True:
            raise BackendError(f"{name} requires backend JSON Schema support")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(properties),
                    "properties": properties,
                },
            },
        }

    def generate_speech_candidates(self, pre, *, known_wolf_ids, q):
        """使用同一 PRE 的冻结 Q，一次生成有序的完整发言候选。"""
        from werewolf.observer_dtrs.speech_candidates import (
            SPEECH_CANDIDATE_PROMPT,
            build_candidate_context,
            candidate_response_properties,
            parse_speech_candidates,
        )

        context = build_candidate_context(pre, known_wolf_ids, q)
        prompt = SPEECH_CANDIDATE_PROMPT + "\n\n" + json.dumps(context, ensure_ascii=False)
        temperature, max_tokens = self._request_limits()
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=self._decision_response_format(
                "speech_candidates", **candidate_response_properties(context),
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if metadata["finish_reason"] == "length":
            raise GameplayActionValidationError("发言候选响应被截断")
        return parse_speech_candidates(content, pre=pre, context=context)

    def verify_speech_candidate_bases(self, batch, *, known_wolf_ids):
        """一次核验整批候选的预测前提及公开引用。"""
        from werewolf.observer_dtrs.action_selection import (
            BASIS_VERIFICATION_PROMPT, basis_verification_context, basis_verification_properties, parse_basis_verification,
        )

        context = basis_verification_context(batch, known_wolf_ids)
        temperature, max_tokens = self._request_limits()
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": BASIS_VERIFICATION_PROMPT + "\n\n" + json.dumps(context, ensure_ascii=False)}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=self._decision_response_format(
                "speech_candidate_bases", **basis_verification_properties(context),
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if metadata["finish_reason"] == "length":
            raise GameplayActionValidationError("候选依据核验响应被截断")
        return parse_basis_verification(content, batch, context)

    def select_report_event_plan(self, context, *, basis_refs):
        """在私有策略调用中选一个合法发言计划，返回索引和显式依据。"""
        candidate_indices = list(range(len(context["candidates"])))
        prompt = (
            "你是当前发言的狼人。最终目标是狼队获胜。请从候选表中选一个发言计划，索引从 0 开始。\n"
            "考虑本轮投票及可能的 PK 给存活狼队带来的放逐威胁、公开证据、发言可信度、队友利益和后续策略空间。"
            "如果有利于狼队最终获胜，可以承担眼前的嫌疑或放逐风险。\n"
            "ACCUSE_WOLF 为指狼，CLEAR 为认好，SUPPORT 和 OPPOSE 为支持或反对目标的公开观点，"
            "SELF_DEFEND 为自辩，NO_COMMITMENT 为不作明确表态。\n"
            "公开发言记录表明玩家说过什么，其中的主张可能失实。report_predictions 中的 probability "
            "预测该观察者此时成功取得的原始嫌疑报告是否包含你，私人报告内容仍未知。\n"
            "adoptable_directions 中的 POS 表示预测报告包含你，NEG 表示预测报告不包含你。"
            "只有一个方向时，该方向是当前明确定下的认知判断。有两个方向时由你判断是否采用。"
            "空列表表示 BND，应保留两种可能。方向权限只约束决策依据，所有候选均可考虑。"
            "例如，已有公开指控就可以基于该指控自辩。\n"
            "登记一个决定性依据。若采用明确预测，引用 prediction:<观察者席位>:<方向>，方向须在该席位列表中。"
            "若依据某个公开事件，引用 public:<event_id>。independent 表示无需采用报告预测方向，"
            "keep_both 表示兼顾两种报告结果。这两种依据始终可用。\n"
            "仅返回 candidate_index 和 basis_ref。"
        )
        temperature, max_tokens = self._request_limits()
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt + "\n\n" + json.dumps(context, ensure_ascii=False)}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=self._decision_response_format(
                "report_event_plan",
                candidate_index={"type": "integer", "enum": candidate_indices},
                basis_ref={"type": "string", "enum": list(basis_refs)},
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if metadata["finish_reason"] == "length":
            raise GameplayActionValidationError("发言计划选择响应被截断")
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise GameplayActionValidationError("发言计划选择响应必须为 JSON") from exc
        if (
            type(payload) is not dict
            or set(payload) != {"candidate_index", "basis_ref"}
            or type(payload["candidate_index"]) is not int
            or payload["candidate_index"] not in candidate_indices
            or type(payload["basis_ref"]) is not str
            or payload["basis_ref"] not in basis_refs
        ):
            raise GameplayActionValidationError("发言计划选择须使用合法候选索引和相容的依据引用")
        return payload["candidate_index"], payload["basis_ref"]

    def _select_discussion_tactic(self, observation, prompt, *, temperature, max_tokens, attempt):
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={
                "stage": "discussion_tactic",
                "observation": observation,
                "gen_times": attempt - 1,
            },
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=self._decision_response_format(
                "discussion_tactic", strategy={"type": "string", "enum": list(DISCUSSION_TACTICS)},
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if metadata["finish_reason"] == "length":
            raise GameplayActionValidationError("discussion tactic response was truncated")
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise GameplayActionValidationError(
                "discussion tactic response is not valid JSON"
            ) from exc
        if (
            not isinstance(payload, dict)
            or set(payload) != {"strategy"}
            or not isinstance(payload["strategy"], str)
            or payload["strategy"] not in DISCUSSION_TACTICS
        ):
            raise GameplayActionValidationError("response must select exactly one legal discussion tactic")
        return payload["strategy"]

    def _generate_day_cognition(
        self,
        observation,
        *,
        pre_speech_belief,
        tom_context=None,
        temperature,
        max_tokens,
    ):
        return self._retry_validated_generation(
            stage="day_cognition",
            generate=lambda attempt, _last_error: self._generate_day_cognition_once(
                observation,
                pre_speech_belief=pre_speech_belief,
                tom_context=tom_context,
                temperature=temperature,
                max_tokens=max_tokens,
                attempt=attempt,
            ),
        )

    def _generate_day_cognition_once(
        self,
        observation,
        *,
        pre_speech_belief,
        tom_context=None,
        temperature,
        max_tokens,
        attempt,
    ):
        player_id = observation.get("current_act_idx")
        phase = observation.get("phase")
        candidate_snapshot = freeze_discussion_candidates(observation)
        claim_catalog = build_public_claim_catalog(observation)
        claim_ids = tuple(claim.claim_id for claim in claim_catalog)
        prompt = build_day_cognition_prompt(
            observation,
            candidate_snapshot=candidate_snapshot,
            claim_catalog=claim_catalog,
            pre_speech_belief=pre_speech_belief,
            tom_context=tom_context,
        )
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={
                "stage": "day_cognition",
                "observation": observation,
                "gen_times": attempt - 1,
            },
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=day_cognition_response_format(
                supports_json_schema=getattr(
                    self.backend,
                    "supports_json_schema",
                    False,
                ),
                candidate_snapshot=candidate_snapshot,
                claim_ids=claim_ids,
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if metadata["finish_reason"] == "length":
            raise BeliefValidationError(
                f"Day cognition response was truncated "
                f"(player={player_id}, phase={phase!r})"
            )
        report = parse_day_cognition_response(
            content,
            player_id=player_id,
            phase=phase,
            candidate_snapshot=candidate_snapshot,
            claim_ids=claim_ids,
        )
        return report, candidate_snapshot, claim_catalog

    def _generate_public_speech(
        self,
        observation,
        *,
        discussion_acts=(),
        claim_catalog=(),
        temperature,
        max_tokens,
        direct_prompt=None,
        multimind_belief=None,
    ):
        """生成并校验提交给环境的公开发言。"""

        return self._retry_validated_generation(
            stage="speech_realization",
            generate=lambda attempt, last_error: self._generate_public_speech_once(
                observation,
                discussion_acts=discussion_acts,
                claim_catalog=claim_catalog,
                temperature=temperature,
                max_tokens=max_tokens,
                attempt=attempt,
                validation_error=last_error,
                direct_prompt=direct_prompt,
                multimind_belief=multimind_belief,
            ),
        )

    def _generate_public_speech_once(
        self,
        observation,
        *,
        discussion_acts,
        claim_catalog,
        temperature,
        max_tokens,
        attempt,
        validation_error,
        direct_prompt,
        multimind_belief=None,
    ):
        phase = observation.get("phase")
        player_id = observation.get("current_act_idx")
        request_options = {}
        if direct_prompt is None:
            prompt = build_public_speech_realization_prompt(
                observation,
                discussion_acts=discussion_acts,
                claim_catalog=claim_catalog,
                multimind_belief=multimind_belief,
            )
            retry_instruction = "必须基于同一组冻结意图完整重新生成一段发言。"
        else:
            prompt = direct_prompt
            retry_instruction = "按照同一请求重新生成内部推理和公开发言。"
            request_options["response_format"] = self._decision_response_format(
                "direct_gameplay_speech", thought={"type": "string"}, speech={"type": "string"},
            )
        if validation_error is not None:
            prompt += (
                "\n\n上一次输出未通过本地严格验证：\n"
                f"{validation_error}\n\n"
                f"{retry_instruction}"
                "不得修改、引用或解释上一次响应。"
            )
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={
                "stage": "speech_realization",
                "observation": observation,
                "gen_times": attempt - 1,
            },
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **request_options,
        )
        if direct_prompt is not None and metadata["finish_reason"] != "length":
            try:
                speech_response = json.loads(content)
            except (TypeError, json.JSONDecodeError) as exc:
                raise GameplaySpeechQualityError("direct speech response is not valid JSON") from exc
            if (
                not isinstance(speech_response, dict)
                or set(speech_response) != {"thought", "speech"}
                or not isinstance(speech_response["thought"], str)
                or not speech_response["thought"].strip()
            ):
                raise GameplaySpeechQualityError("direct speech requires thought and speech fields")
            content = speech_response["speech"]
        validate_gameplay_public_speech(
            content,
            finish_reason=metadata["finish_reason"],
            player_id=player_id,
            phase=phase,
            # MultiMind 的计划遵从由提交后的 V1 标注衡量。原冻结意图继续校验目标。
            required_player_ids=() if multimind_belief is not None else (
                act.target
                for act in discussion_acts
                if act.target is not None
            ),
        )
        return content.strip()

    def _generate_vote(
        self,
        observation,
        *,
        belief,
        temperature,
        max_tokens,
    ):
        return self._retry_validated_generation(
            stage="vote",
            generate=lambda attempt, _last_error: self._generate_vote_once(
                observation,
                belief=belief,
                temperature=temperature,
                max_tokens=max_tokens,
                attempt=attempt,
            ),
        )

    def _generate_vote_once(
        self,
        observation,
        *,
        belief,
        temperature,
        max_tokens,
        attempt,
    ):
        phase = observation.get("phase")
        candidates = self.freeze_legal_vote_candidates(
            observation["valid_action"],
            phase=phase,
        )
        legal_targets = tuple(target for target, _action in candidates)
        action_by_target = dict(candidates)
        belief_payload = None
        if belief is not None:
            belief_payload = (
                belief.gameplay_dict() if self.decision_method is None else belief.as_dict()
            )
        prompt = build_vote_prompt(observation, belief_payload, legal_targets)
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={
                "stage": "vote",
                "observation": observation,
                "gen_times": attempt - 1,
            },
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=vote_response_format(
                supports_json_schema=getattr(self.backend, "supports_json_schema", False),
                legal_targets=legal_targets,
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if metadata["finish_reason"] == "length":
            raise GameplayActionValidationError(
                f"vote response was truncated (phase={phase!r})"
            )
        target = parse_vote_response(content, legal_targets=legal_targets, phase=phase)
        return action_by_target[target]

    def _generate_night_action(
        self,
        observation,
        *,
        temperature,
        max_tokens,
    ):
        return self._retry_validated_generation(
            stage="night_action",
            generate=lambda attempt, _last_error: self._generate_night_action_once(
                observation,
                temperature=temperature,
                max_tokens=max_tokens,
                attempt=attempt,
            ),
        )

    def _generate_night_action_once(
        self,
        observation,
        *,
        temperature,
        max_tokens,
        attempt,
    ):
        phase = observation.get("phase")
        candidate_snapshot = self.freeze_authoritative_action_candidates(
            observation["valid_action"]
        )
        prompt = self.format_observation(observation, action_candidates=candidate_snapshot)
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={
                "stage": "night",
                "observation": observation,
                "gen_times": attempt - 1,
            },
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=night_action_response_format(
                supports_json_schema=getattr(self.backend, "supports_json_schema", False),
                candidate_snapshot=candidate_snapshot,
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if metadata["finish_reason"] == "length":
            raise GameplayActionValidationError(
                f"night action response was truncated (phase={phase!r}, finish_reason='length')"
            )
        _action, env_action = self.parse_night_action_selection(
            content,
            candidate_snapshot,
            phase=phase,
        )
        return env_action
