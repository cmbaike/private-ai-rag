from types import SimpleNamespace
from unittest.mock import patch

from app.groundedness import (
    assess_groundedness,
    parse_groundedness_response,
)


def response(
    content,
    finish_reason="stop",
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                ),
            )
        ]
    )


def test_non_stop_fails_closed():
    result = parse_groundedness_response(
        '{"grounded": true, "reason": "ok"}',
        "length",
    )

    assert result.grounded is False


def test_empty_content_fails_closed():
    result = parse_groundedness_response(
        "",
        "stop",
    )

    assert result.grounded is False


def test_invalid_json_fails_closed():
    result = parse_groundedness_response(
        "not json",
        "stop",
    )

    assert result.grounded is False


def test_valid_result_is_parsed():
    result = parse_groundedness_response(
        '{"grounded": true, '
        '"reason": "Directly stated."}',
        "stop",
    )

    assert result.grounded is True
    assert result.reason == "Directly stated."
    
def test_non_stop_preserves_diagnostics():
    result = parse_groundedness_response(
        '{"grounded": true, "reason": "ok"}',
        "length",
        500,
    )

    assert result.grounded is False
    assert result.judge_error is True
    assert result.finish_reason == "length"
    assert result.completion_tokens == 500      


@patch(
    "app.groundedness.client.chat.completions.create"
)
def test_think_block_is_removed(create):
    create.return_value = response(
        "<think>internal reasoning</think>\n"
        '{"grounded": true, '
        '"reason": "Evidence supports the claim."}'
    )

    result = assess_groundedness(
        "claim",
        "evidence",
    )

    assert result.grounded is True
    assert (
        result.reason
        == "Evidence supports the claim."
    )
    
    
  
