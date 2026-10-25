"""Only local validation is exposed until the next research approval."""



def register(commands):
    check = commands.add_parser("check-llm-reasoner-config", help="validate frozen LLM policy; no model/backend is loaded")
    check.add_argument("--config", required=True)


def check_config(path):
    from werewolf.tom_llm.common import digest
    from werewolf.tom_llm.policy import load_policy, policy_binding
    config = load_policy(path)
    return {"status": "valid", "config_digest": digest(config), "binding": policy_binding(),
            "execution": "local_validation_only", "formal_readiness": False}
