"""
behavior_utils.py

Shared utilities for running an autonomous behavior loop: repeatedly read
the camera, classify the current frame, and let a student-authored
decision function decide what the robot should do next.

Why a loop at all: the RVR's firmware stops the robot automatically if it
doesn't receive a new drive command within about 2 seconds. So driving
continuously -- not just for one instant -- requires re-issuing a drive
command faster than that timeout, over and over, for as long as the robot
should keep moving. start_behavior_loop() is exactly that loop: it calls
the decision function repeatedly, on a fixed schedule well under 2 seconds,
so "keep driving" is a natural side effect of the loop running rather than
something a student has to build themselves.

Why a dedicated thread, and why Start/Stop buttons: this mirrors the same
control-loop pattern used in gamepad_utils.py. Running everything on a
single dedicated thread -- rather than reacting directly from, say, a
camera callback -- keeps behavior predictable and makes an immediate stop
possible at any time via the Stop button, regardless of what the loop is
doing. Nothing drives until Start is pressed, even though the setup cell
that defines everything runs first.
"""

import time
import threading
import traceback

import ipywidgets as widgets
from sphero_sdk import RawMotorModesEnum

from inference_utils import predict_image
from jupyter_utils import register_click_handler

# Module-level state, so start/stop are idempotent across cell re-runs and
# repeated button clicks -- mirrors the pattern in gamepad_utils.py.
_behavior_thread = None
_behavior_loop_running = False


def start_behavior_loop(rvr, camera, model, class_names, device, decision_fn, poll_interval=0.5):
    """Starts a dedicated thread that repeatedly reads the camera,
    classifies the current frame, and calls decision_fn(rvr, label) to
    decide what the robot should do next.

    poll_interval must stay comfortably under the RVR's ~2 second
    command-timeout window -- the default of 0.5s means decision_fn gets
    called, and can re-issue a drive command, about twice a second, so a
    "keep driving" decision never goes stale enough for the RVR's own
    safety timeout to kick in.

    Idempotent: stops any previously running behavior loop first, so
    clicking Start twice (or re-running a setup cell) doesn't spawn a
    second thread fighting over the same rvr.

    Args:
        rvr: from robot_utils.get_rvr()
        camera (TraitletCamera): the live camera feed, already started
        model (torch.nn.Module): a loaded, eval-mode model
        class_names (list[str]): class names in the model's output order
        device: torch.device the model lives on
        decision_fn (callable): called every iteration as
            decision_fn(rvr, label) -- this is where the actual behavior
            logic lives, and is expected to command the robot directly
            (e.g. rvr.raw_motors(...), rvr.drive_with_heading(...))
        poll_interval (float): seconds between iterations; keep well
            under 2 seconds (the RVR's own command timeout)

    Returns:
        None
    """
    global _behavior_thread, _behavior_loop_running

    stop_behavior_loop()

    def behavior_loop():
        global _behavior_loop_running
        try:
            while _behavior_loop_running:
                try:
                    frame = camera.value
                    label, confidence = predict_image(model, frame, class_names, device)
                    decision_fn(rvr, label)
                except Exception:
                    # Print any error immediately -- exceptions raised on a
                    # background thread inside Jupyter can otherwise
                    # disappear without a trace. A bug in decision_fn
                    # shouldn't silently kill the loop with no explanation.
                    traceback.print_exc()

                time.sleep(poll_interval)
        finally:
            # Safety net: no matter how/why the loop exits, always leave
            # the RVR with an explicit zero-speed command rather than
            # relying on its ~2 second stale-command watchdog.
            try:
                rvr.raw_motors(
                    left_mode=RawMotorModesEnum.forward.value, left_duty_cycle=0,
                    right_mode=RawMotorModesEnum.forward.value, right_duty_cycle=0,
                )
                print('Safety net: sent explicit zero-speed command on behavior loop exit.')
            except Exception:
                traceback.print_exc()

    _behavior_loop_running = True
    _behavior_thread = threading.Thread(name='behavior_loop_thread', target=behavior_loop, daemon=True)
    _behavior_thread.start()
    print('Autonomous behavior started.')


def stop_behavior_loop():
    """Stops the behavior loop started by start_behavior_loop(), if one is
    running. Safe to call even if no loop is running.

    Returns:
        None
    """
    global _behavior_loop_running, _behavior_thread

    if _behavior_thread is not None and _behavior_thread.is_alive():
        _behavior_loop_running = False
        _behavior_thread.join(timeout=1)
        print('Autonomous behavior stopped.')

    _behavior_thread = None


def create_start_stop_buttons(rvr, camera, model, class_names, device, decision_fn, poll_interval=0.5):
    """Builds a large green Start button and a large red Stop button that
    begin/end an autonomous behavior loop.

    Nothing drives until Start is pressed, even though the cell that calls
    this function (and sets everything up) has already run -- driving only
    begins on the button click, and Stop is always clickable regardless of
    what the behavior loop is currently doing.

    Idempotent: safe to re-run the cell that calls this -- register_click_handler
    ensures re-running doesn't stack duplicate button handlers.

    Args:
        rvr: from robot_utils.get_rvr()
        camera (TraitletCamera): the live camera feed, already started
        model (torch.nn.Module): a loaded, eval-mode model
        class_names (list[str]): class names in the model's output order
        device: torch.device the model lives on
        decision_fn (callable): decision_fn(rvr, label) -- see
            start_behavior_loop()
        poll_interval (float): seconds between loop iterations

    Returns:
        widgets.HBox: the Start/Stop button pair, ready to display()
    """
    button_layout = widgets.Layout(width='200px', height='100px')
    start_button = widgets.Button(description='Start', button_style='success', layout=button_layout)
    stop_button = widgets.Button(description='Stop', button_style='danger', layout=button_layout)

    def on_start_clicked(button):
        start_behavior_loop(rvr, camera, model, class_names, device, decision_fn, poll_interval)

    def on_stop_clicked(button):
        stop_behavior_loop()

    register_click_handler(start_button, on_start_clicked)
    register_click_handler(stop_button, on_stop_clicked)

    return widgets.HBox([start_button, stop_button])
