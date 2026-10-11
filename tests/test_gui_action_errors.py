import gradio as gr
import pytest

from musubi_tuner_gui.common_gui import run_gui_training_action


def test_gui_training_validation_error_does_not_claim_training_started(monkeypatch):
    notices = []
    monkeypatch.setattr(gr, "Info", notices.append)

    def fail_validation():
        raise ValueError("synthetic invalid preset")

    with pytest.raises(gr.Error, match="synthetic invalid preset") as error:
        run_gui_training_action(
            fail_validation,
            display_name="Test Model",
            print_only=False,
        )

    assert notices == []
    assert error.value.title == "Test Model training could not start"


@pytest.mark.parametrize(
    ("print_only", "message"),
    [
        (False, "Training started. Please check the console for progress."),
        (True, "Training command preview generated. Check the console/log for the full command."),
    ],
)
def test_gui_training_success_notice_is_emitted_after_callback(monkeypatch, print_only, message):
    events = []
    monkeypatch.setattr(gr, "Info", lambda text: events.append(("notice", text)))

    result = run_gui_training_action(
        lambda: events.append(("callback", "complete")) or "result",
        display_name="Test Model",
        print_only=print_only,
    )

    assert result == "result"
    assert events == [("callback", "complete"), ("notice", message)]


def test_gui_training_suppresses_started_notice_when_cancelled_during_start(monkeypatch):
    notices = []
    monkeypatch.setattr(gr, "Info", notices.append)

    result = run_gui_training_action(
        lambda: "result",
        display_name="Test Model",
        print_only=False,
        is_running=lambda: False,
    )

    assert result == "result"
    assert notices == []
