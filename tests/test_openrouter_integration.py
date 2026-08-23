from types import SimpleNamespace

from benchmarks.benchmark import get_model
from benchmarks.models.openrouter_model import OpenRouterModel
from src.openrouter import OpenRouterClient


class FlakyCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary OpenRouter failure")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"invoice_number": "INV-123"}')
                )
            ],
            usage=None,
        )


def test_benchmark_openrouter_registered():
    model = get_model("openrouter", {})

    assert isinstance(model, OpenRouterModel)
    assert model.model_name == "qwen/qwen2.5-vl-72b-instruct"


def test_benchmark_openrouter_uses_model_path_as_model_id():
    model = get_model("openrouter", {"model_path": "qwen/qwen3-vl-235b-a22b-instruct"})

    assert isinstance(model, OpenRouterModel)
    assert model.model_name == "qwen/qwen3-vl-235b-a22b-instruct"


def test_benchmark_openrouter_parses_json_invoice_number():
    model = OpenRouterModel()

    assert model._clean_output('{"invoice_number": "INV-123"}') == "INV-123"


def test_benchmark_openrouter_result_method_is_model_name():
    model = OpenRouterModel({"model_path": "qwen/qwen3-vl-8b-instruct"})
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"invoice_number": "INV-123"}')
            )
        ],
        usage=None,
    )
    completions = SimpleNamespace(create=lambda *args, **kwargs: response)
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = model.predict(text="invoice number INV-123")

    assert result.method == "qwen/qwen3-vl-8b-instruct"


def test_benchmark_openrouter_retries_transient_failures():
    model = OpenRouterModel(
        {
            "model_path": "qwen/qwen3-vl-8b-instruct",
            "max_retries": 1,
            "retry_backoff_seconds": 0,
        }
    )
    completions = FlakyCompletions()
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = model.predict(text="invoice number INV-123")

    assert completions.calls == 2
    assert result.invoice_number == "INV-123"
    assert result.metadata["retry_count"] == 1


def test_benchmark_openrouter_parses_null_invoice_number():
    model = OpenRouterModel()

    assert model._clean_output('{"invoice_number": null}') is None


def test_benchmark_openrouter_rejects_malformed_responses():
    model = OpenRouterModel()

    assert model._clean_output("I found invoice INV-123") is None
    assert model._clean_output('"INV-123"') is None
    assert model._clean_output('["INV-123"]') is None
    assert model._clean_output('{"invoice_number": 12345}') is None
    assert (
        model._clean_output('{"invoice_number": "The invoice number is INV-123."}')
        is None
    )
    assert model._clean_output('{"invoice_number": "Invoice INV-123"}') is None
    assert model._clean_output('{"invoice_number": "INV-123 extracted"}') is None
    assert (
        model._clean_output(
            '{"invoice_number": "{\\"invoice_number\\": \\"INV-123\\"}"}'
        )
        is None
    )
    assert (
        model._clean_output(
            'Here is the result:\n```json\n{"invoice_number": "INV-123"}\n```'
        )
        is None
    )


def test_fallback_openrouter_parses_json_invoice_number():
    client = OpenRouterClient()

    assert client._clean_output('{"invoice_number": "CS-991"}') == "CS-991"


def test_fallback_openrouter_rejects_malformed_responses():
    client = OpenRouterClient()

    assert client._clean_output("I found invoice INV-123") is None
    assert client._clean_output('"INV-123"') is None
    assert client._clean_output('["INV-123"]') is None
    assert client._clean_output('{"invoice_number": 12345}') is None
    assert (
        client._clean_output(
            '{"invoice_number": "Sorry, I cannot determine the invoice number."}'
        )
        is None
    )
    assert client._clean_output('{"invoice_number": "Invoice CS-991"}') is None
    assert client._clean_output('{"invoice_number": "CS-991 extracted"}') is None
    assert (
        client._clean_output(
            '{"invoice_number": "{\\"invoice_number\\": \\"CS-991\\"}"}'
        )
        is None
    )
    assert (
        client._clean_output('Sure:\n```json\n{"invoice_number": "CS-991"}\n```')
        is None
    )


def test_fallback_openrouter_retries_transient_failures():
    client = OpenRouterClient(api_key="test", model_name="qwen/qwen3-vl-8b-instruct")
    completions = FlakyCompletions()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._initialized = True
    client.max_retries = 1
    client.retry_backoff_seconds = 0

    result = client.predict(words=["invoice", "number", "INV-123"])

    assert completions.calls == 2
    assert result["invoice_number"] == "INV-123"
    assert result["method"] == "qwen/qwen3-vl-8b-instruct"
    assert result["retry_count"] == 1
