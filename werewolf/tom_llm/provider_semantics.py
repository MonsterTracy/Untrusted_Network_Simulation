"""Failure classification seam only. No HTTP client, live runner or admission factory."""

import httpx2 as httpx
import openai

from werewolf.tom_llm.inference import InfrastructureFailure, TransportFailure


class NotExecuted(TransportFailure):
    """Explicit rejection or failure before connecting; safe within dispatch budget."""


class OutcomeUnknown(InfrastructureFailure):
    """May have reached the service. Never automatically resend or invent a reply."""


def classify_provider_error(error):
    if isinstance(error, openai.RateLimitError) and getattr(error, 'status_code', None) == 429:
        return NotExecuted('explicit HTTP 429 rejection')
    cause = error.__cause__
    if isinstance(error, openai.APIConnectionError) and isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout)):
        return NotExecuted('failed before connection was established')
    if isinstance(error, (openai.APIConnectionError, httpx.ReadTimeout, httpx.WriteTimeout,
                          httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)):
        return OutcomeUnknown('request acceptance/outcome unknown; quarantine without retry')
    return InfrastructureFailure('provider infrastructure failure; no uniform fallback')


def require_live_admission(*_args, **_kwargs):
    raise NotImplementedError('physical Admission and live Execution are not implemented; CPU/Mock is insufficient')
