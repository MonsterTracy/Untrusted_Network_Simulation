"""Static capability diagnostic; full runtime preflight is owned by Pilot-T CLI."""


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    from werewolf.interventions.online import preflight as phase2_online_preflight
    return getattr(phase2_online_preflight, name)


def main():
    from werewolf.interventions.online.preflight import main as static_main
    return static_main()


if __name__ == "__main__":
    raise SystemExit(main())
