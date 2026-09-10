import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
import openai

try:
    from werewolf.backends import (
        BackendError,
        LLMBackend,
        OpenAICompatibleBackend,
    )
    from werewolf.backends.openai_compatible import (
        _is_loopback_base_url,
    )
except ModuleNotFoundError:
    BackendError = None
    LLMBackend = None
    OpenAICompatibleBackend = None
    _is_loopback_base_url = None


class BackendAvailabilityTest(unittest.TestCase):
    def test_backend_package_is_available(self):
        self.assertIsNotNone(LLMBackend)


class FakeCompletions:
    def __init__(
        self,
        content="backend response",
        error=None,
        usage=None,
        finish_reason=None,
    ):
        self.content = content
        self.error = error
        self.usage = usage
        self.finish_reason = finish_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message,
                    finish_reason=self.finish_reason,
                )
            ],
            usage=self.usage,
            id="provider-response-id",
        )


class FakeClient:
    def __init__(
        self,
        content="backend response",
        error=None,
        usage=None,
        finish_reason=None,
    ):
        self.completions = FakeCompletions(
            content=content,
            error=error,
            usage=usage,
            finish_reason=finish_reason,
        )
        self.chat = SimpleNamespace(completions=self.completions)


@unittest.skipIf(LLMBackend is None, "backend package is not implemented")
class BackendTest(unittest.TestCase):
    def test_loopback_base_url_detection_uses_parsed_hostname(self):
        loopback_urls = (
            "http://127.0.0.1:8080/v1",
            "http://127.0.0.2:8080/v1",
            "http://localhost:8080/v1",
            "http://LOCALHOST:8080/v1",
            "http://[::1]:8080/v1",
        )
        non_loopback_urls = (
            "http://127.0.0.1.example.com/v1",
            "http://localhost.example.com/v1",
            "http://example.com/v1",
            "https://api.openai.com/v1",
            "https://api.deepseek.com/v1",
            "http://10.0.0.5/v1",
            "http://172.16.0.5/v1",
            "http://192.168.1.5/v1",
            "not-a-url",
            None,
        )

        for base_url in loopback_urls:
            with self.subTest(base_url=base_url):
                self.assertTrue(_is_loopback_base_url(base_url))
        for base_url in non_loopback_urls:
            with self.subTest(base_url=base_url):
                self.assertFalse(_is_loopback_base_url(base_url))

    @patch("werewolf.backends.openai_compatible.openai.OpenAI")
    @patch("werewolf.backends.openai_compatible.openai.DefaultHttpx2Client")
    def test_loopback_backend_disables_environment_proxy_only(
        self,
        http_client_class,
        openai_client_class,
    ):
        loopback_client = object()
        http_client_class.return_value = loopback_client

        for base_url in (
            "http://127.0.0.1:8080/v1",
            "http://127.0.0.2:8080/v1",
            "http://localhost:8080/v1",
            "http://LOCALHOST:8080/v1",
            "http://[::1]:8080/v1",
        ):
            with self.subTest(base_url=base_url):
                http_client_class.reset_mock()
                openai_client_class.reset_mock()
                backend = OpenAICompatibleBackend(
                    api_key="local-mlx",
                    base_url=base_url,
                    default_model="local-model",
                    max_retries=0,
                    supports_json_schema=True,
                )

                self.assertEqual(backend.default_model, "local-model")
                self.assertTrue(backend.supports_json_schema)
                http_client_class.assert_called_once_with(
                    trust_env=False,
                )
                openai_client_class.assert_called_once_with(
                    api_key="local-mlx",
                    base_url=base_url,
                    max_retries=0,
                    http_client=loopback_client,
                )

    @patch("werewolf.backends.openai_compatible.openai.OpenAI")
    @patch("werewolf.backends.openai_compatible.openai.DefaultHttpx2Client")
    def test_remote_backend_keeps_default_http_client_behavior(
        self,
        http_client_class,
        openai_client_class,
    ):
        for base_url in (
            "http://127.0.0.1.example.com/v1",
            "http://localhost.example.com/v1",
            "https://api.openai.com/v1",
            "https://api.deepseek.com/v1",
            "http://10.0.0.5/v1",
            "http://172.16.0.5/v1",
            "http://192.168.1.5/v1",
        ):
            with self.subTest(base_url=base_url):
                openai_client_class.reset_mock()
                OpenAICompatibleBackend(
                    api_key="secret",
                    base_url=base_url,
                    default_model="remote-model",
                    max_retries=0,
                )
                openai_client_class.assert_called_once_with(
                    api_key="secret",
                    base_url=base_url,
                    max_retries=0,
                )

        http_client_class.assert_not_called()

    @patch("werewolf.backends.openai_compatible.openai.OpenAI")
    def test_loopback_native_client_preserves_sdk_timeout_family(
        self, openai_client_class
    ):
        OpenAICompatibleBackend(
            api_key="local-test",
            base_url="http://127.0.0.1:8000/v1",
            max_retries=0,
        )
        transport = openai_client_class.call_args.kwargs["http_client"]
        try:
            self.assertIsInstance(transport, httpx2.Client)
            self.assertIsInstance(transport.timeout, httpx2.Timeout)
            self.assertEqual(transport.timeout, openai.DEFAULT_TIMEOUT)
            self.assertTrue(transport.follow_redirects)
            self.assertEqual(openai_client_class.call_args.kwargs["max_retries"], 0)
        finally:
            transport.close()

    @patch("werewolf.backends.openai_compatible.openai.OpenAI")
    @patch("werewolf.backends.openai_compatible.openai.DefaultHttpx2Client")
    def test_injected_loopback_client_bypasses_transport_construction(
        self, http_client_class, openai_client_class
    ):
        client = FakeClient()
        backend = OpenAICompatibleBackend(
            base_url="http://[::1]:8000/v1",
            client=client,
            supports_json_schema=True,
            max_retries=0,
        )
        self.assertIs(backend.client, client)
        self.assertTrue(backend.supports_json_schema)
        http_client_class.assert_not_called()
        openai_client_class.assert_not_called()

    def test_fake_backend_can_implement_common_interface(self):
        class FakeBackend(LLMBackend):
            def chat(
                self,
                messages,
                model=None,
                temperature=0.7,
                max_tokens=None,
                response_format=None,
                **kwargs,
            ):
                return "fake response"

        self.assertEqual(FakeBackend().chat([{"role": "user", "content": "hi"}]), "fake response")

    def test_openai_backend_uses_injected_client_without_api_key(self):
        client = FakeClient(content="model output")
        backend = OpenAICompatibleBackend(
            api_key=None,
            default_model="default-model",
            client=client,
        )

        result = backend.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="override-model",
            temperature=0.2,
            max_tokens=123,
            response_format={"type": "json_object"},
            seed=7,
        )

        self.assertEqual(result, "model output")
        self.assertEqual(
            client.completions.calls,
            [
                {
                    "model": "override-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.2,
                    "max_tokens": 123,
                    "response_format": {"type": "json_object"},
                    "seed": 7,
                }
            ],
        )

    def test_json_schema_capability_is_explicit_and_default_off(self):
        default_backend = OpenAICompatibleBackend(
            client=FakeClient(),
        )
        schema_backend = OpenAICompatibleBackend(
            client=FakeClient(),
            supports_json_schema=True,
        )

        self.assertFalse(default_backend.supports_json_schema)
        self.assertTrue(schema_backend.supports_json_schema)
        with self.assertRaises(TypeError):
            OpenAICompatibleBackend(
                client=FakeClient(),
                supports_json_schema="true",
            )

    def test_openai_backend_omits_none_optional_parameters(self):
        client = FakeClient()
        backend = OpenAICompatibleBackend(client=client, default_model="model")

        backend.chat(
            messages=[{"role": "user", "content": "hello"}],
            temperature=None,
        )

        self.assertEqual(
            client.completions.calls[0],
            {
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    def test_openai_backend_exposes_usage_without_response_id(self):
        usage = SimpleNamespace(
            prompt_tokens=9,
            completion_tokens=4,
            total_tokens=13,
        )
        backend = OpenAICompatibleBackend(
            client=FakeClient(content="output", usage=usage),
            default_model="model",
        )

        content, normalized_usage = backend.chat_with_metadata(
            messages=[{"role": "user", "content": "hello"}],
        )

        self.assertEqual(content, "output")
        self.assertEqual(
            normalized_usage,
            {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13},
        )
        self.assertNotIn("id", normalized_usage)

    def test_openai_backend_exposes_stop_and_length_finish_reasons(self):
        for finish_reason in ("stop", "length"):
            with self.subTest(finish_reason=finish_reason):
                backend = OpenAICompatibleBackend(
                    client=FakeClient(
                        content="output",
                        finish_reason=finish_reason,
                    ),
                    default_model="model",
                )

                content, metadata = backend.chat_with_metadata(
                    messages=[{"role": "user", "content": "hello"}],
                )

                self.assertEqual(content, "output")
                self.assertEqual(
                    metadata,
                    {"finish_reason": finish_reason},
                )

    @patch("werewolf.backends.openai_compatible.openai.OpenAI")
    def test_openai_backend_can_disable_transport_retries(self, client_class):
        OpenAICompatibleBackend(
            api_key="secret",
            default_model="model",
            max_retries=0,
        )

        client_class.assert_called_once_with(
            api_key="secret",
            max_retries=0,
        )

    def test_openai_backend_requires_model(self):
        backend = OpenAICompatibleBackend(client=FakeClient())

        with self.assertRaises(BackendError):
            backend.chat(messages=[])

    def test_openai_backend_wraps_provider_errors(self):
        backend = OpenAICompatibleBackend(
            client=FakeClient(error=RuntimeError("provider failed")),
            default_model="model",
        )

        with self.assertRaises(BackendError) as raised:
            backend.chat(messages=[])

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_openai_backend_rejects_non_text_content(self):
        backend = OpenAICompatibleBackend(
            client=FakeClient(content=None),
            default_model="model",
        )

        with self.assertRaises(BackendError):
            backend.chat(messages=[])






if __name__ == "__main__":
    unittest.main()
