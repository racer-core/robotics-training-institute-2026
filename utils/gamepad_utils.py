"""
gamepad_utils.py

Shared utilities for connecting an Xbox gamepad and driving the RVR from it
using a single dedicated control-loop thread.

Why a dedicated thread: the Gamepad library's own background thread is where
joystick-moved events fire. Calling rvr.raw_motors() or updating ipywidgets
directly from that thread causes two problems -- ipywidgets' display channel
isn't thread-safe (it can freeze), and exceptions raised on a background
thread can fail silently. So the gamepad callbacks here only ever update
plain floats on a GamepadState object; a single control-loop thread (started
by start_control_loop) is the only thing that reads those floats, updates
any UI, and calls rvr.raw_motors() -- at a fixed rate, which also naturally
smooths out motor commands instead of firing one per tiny joystick tick.
"""

import time
import threading
import traceback

import Gamepad
from sphero_sdk import RawMotorModesEnum

# Module-level state, so connect/start/stop are idempotent across cell
# re-runs -- mirroring the singleton pattern in robot_utils.get_rvr().
_gamepad = None
_control_thread = None
_control_loop_running = False


class GamepadState:
    """Holds the latest joystick readings, capture-button requests, and
    live bumper-held state.

    Written by the gamepad library's background thread (via the handlers
    registered in connect_gamepad()), read by the control loop thread
    started by start_control_loop(). Plain attribute assignment is atomic
    under Python's GIL, so no lock is needed for this single-writer/
    single-reader case.

    Attributes:
        left (float): left stick position, -1 to 1
        right (float): right stick position, -1 to 1
        capture_free_requested (bool): set True when the "free" capture
            button is pressed; the control loop's on_update callback is
            expected to check this each cycle and clear it after handling
        capture_blocked_requested (bool): same, for the "blocked" button
        left_bumper_down (bool): True for as long as the left bumper is
            currently held, False otherwise -- for live status displays,
            unlike the one-shot capture_*_requested flags above
        right_bumper_down (bool): same, for the right bumper
    """

    def __init__(self):
        self.left = 0.0
        self.right = 0.0
        self.capture_free_requested = False
        self.capture_blocked_requested = False
        self.left_bumper_down = False
        self.right_bumper_down = False


def connect_gamepad(gamepad_type=Gamepad.XboxONE, joystick_left='LAS -Y',
                     joystick_right='RAS -Y', button_free='LB', button_blocked='RB',
                     invert=True):
    """Connects to the gamepad and starts its background update thread,
    wiring the two joystick axes and two bumpers to update a returned
    GamepadState.

    The left bumper (button_free) and right bumper (button_blocked) act as
    shutter buttons for data collection: pressing one sets a flag on
    GamepadState. Like the joystick handlers, these button handlers only
    ever set a plain flag -- they never touch the RVR, the camera, or any
    widget directly. Whatever's driving the control loop (its on_update
    callback) is expected to check these flags each cycle, act on them,
    and clear them -- keeping all the actual work on that single thread
    instead of the gamepad library's own background thread.

    Button names 'LB'/'RB' are confirmed correct for this project's Xbox
    One controller mapping.

    Idempotent: if a gamepad is already connected (e.g. this cell was
    re-run), it's disconnected first so we don't end up with two
    competing background update threads.

    Args:
        gamepad_type: a Gamepad subclass, e.g. Gamepad.XboxONE
        joystick_left (str): axis name for the left stick
        joystick_right (str): axis name for the right stick
        button_free (str): button name that requests a "free" capture
        button_blocked (str): button name that requests a "blocked" capture
        invert (bool): if True (default), flips both joystick axes --
            matches the common convention where pushing the stick "up"
            should drive the robot forward

    Returns:
        (Gamepad, GamepadState): the connected gamepad object, and a
            state object whose fields update live as the gamepad is used
    """
    global _gamepad

    disconnect_gamepad()

    if not Gamepad.available():
        print('Please connect your gamepad...')
        while not Gamepad.available():
            time.sleep(1.0)

    gamepad = gamepad_type()

    # Matching Dev1's proven working order: start background updates BEFORE
    # registering any axis/button handlers. The controller's real axis/button
    # count isn't fully known until background updates begin polling the
    # device -- registering handlers first can hit a stale/incomplete count
    # (surfacing as a "Button N was not found" error for an axis that the
    # controller does actually have).
    gamepad.startBackgroundUpdates()

    state = GamepadState()
    sign = -1 if invert else 1

    def right_axis_moved(position):
        state.right = sign * position

    def left_axis_moved(position):
        state.left = sign * position

    def free_button_pressed():
        state.capture_free_requested = True
        state.left_bumper_down = True

    def free_button_released():
        state.left_bumper_down = False

    def blocked_button_pressed():
        state.capture_blocked_requested = True
        state.right_bumper_down = True

    def blocked_button_released():
        state.right_bumper_down = False

    _register_axis_handler(gamepad, joystick_right, right_axis_moved)
    _register_axis_handler(gamepad, joystick_left, left_axis_moved)
    _register_button_handler(gamepad, button_free, free_button_pressed, event='pressed')
    _register_button_handler(gamepad, button_free, free_button_released, event='released')
    _register_button_handler(gamepad, button_blocked, blocked_button_pressed, event='pressed')
    _register_button_handler(gamepad, button_blocked, blocked_button_released, event='released')

    _gamepad = gamepad

    print('Gamepad connected.')
    print(f"Left bumper ({button_free}) = capture 'free', right bumper ({button_blocked}) = capture 'blocked'")
    return gamepad, state


def _describe_gamepad_capabilities(gamepad):
    """Best-effort summary of what axes/buttons this connected gamepad is
    actually reporting, for diagnostic printouts. Uses getattr() throughout
    since these attribute names aren't guaranteed across Gamepad library
    versions -- this is a "nice to have" debugging aid, not load-bearing.
    """
    axis_names = getattr(gamepad, 'axisNames', None)
    num_axes = len(getattr(gamepad, 'movedEventMap', []))
    button_names = getattr(gamepad, 'buttonNames', None)
    num_buttons = len(getattr(gamepad, 'pressedEventMap', []))

    lines = [f"This controller is currently reporting {num_axes} axes and {num_buttons} buttons."]
    if axis_names:
        lines.append(f"Axis names this mapping expects: {list(axis_names.keys())}")
    if button_names:
        lines.append(f"Button names this mapping expects: {list(button_names.keys())}")
    lines.append(
        "If an axis/button you expected is missing, this usually means the controller "
        "connected over Bluetooth is exposing fewer axes/buttons than a full Xbox One "
        "controller normally has. Try reconnecting the controller, or run `jstest "
        "/dev/input/js0` in a terminal to see what it's actually reporting."
    )
    return "\n".join(lines)


def _register_axis_handler(gamepad, axis_name, callback):
    """Wraps gamepad.addAxisMovedHandler() with a clearer error message if
    the requested axis isn't available on the currently connected
    controller, instead of a bare KeyError/ValueError traceback.
    """
    try:
        gamepad.addAxisMovedHandler(axis_name, callback)
    except ValueError:
        print(f"\nCouldn't register the '{axis_name}' axis.")
        print(_describe_gamepad_capabilities(gamepad))
        raise


def _register_button_handler(gamepad, button_name, callback, event='pressed'):
    """Wraps gamepad.addButtonPressedHandler()/addButtonReleasedHandler()
    with a clearer error message if the requested button isn't available
    on the currently connected controller, instead of a bare
    KeyError/ValueError traceback.

    Args:
        gamepad: the connected Gamepad object
        button_name (str): button name, e.g. 'LB'
        callback (callable): handler to register
        event (str): 'pressed' or 'released'
    """
    try:
        if event == 'pressed':
            gamepad.addButtonPressedHandler(button_name, callback)
        else:
            gamepad.addButtonReleasedHandler(button_name, callback)
    except ValueError:
        print(f"\nCouldn't register the '{button_name}' button ({event}).")
        print(_describe_gamepad_capabilities(gamepad))
        raise


def disconnect_gamepad():
    """Disconnects the gamepad, if one is currently connected. Safe to
    call even if none is connected.

    Returns:
        None
    """
    global _gamepad

    if _gamepad is not None:
        try:
            _gamepad.disconnect()
        except Exception:
            pass
        _gamepad = None


def stick_to_motor_command(stick_value, max_speed=225):
    """Maps a signed joystick axis [-1, 1] to raw_motors() parameters.

    Args:
        stick_value (float): joystick position, -1 to 1
        max_speed (int): duty cycle ceiling, 0-255

    Returns:
        (int, int): (duty_cycle, mode) ready to pass to rvr.raw_motors()
    """
    duty_cycle = int(round(min(abs(stick_value), 1.0) * max_speed))
    mode = RawMotorModesEnum.forward.value if stick_value >= 0 else RawMotorModesEnum.reverse.value
    return duty_cycle, mode


def start_control_loop(rvr, gamepad_state, poll_interval=0.05, max_speed=225, on_update=None):
    """Starts a single dedicated thread that reads `gamepad_state` at a
    fixed rate and drives `rvr` accordingly, for the life of the teleop
    session.

    Idempotent: stops any previously running control loop first, so
    re-running this cell doesn't spawn a second thread fighting over the
    same rvr.

    Args:
        rvr (SpheroRvrObserver): from robot_utils.get_rvr()
        gamepad_state (GamepadState): from connect_gamepad()
        poll_interval (float): seconds between control loop updates
        max_speed (int): duty cycle ceiling passed to stick_to_motor_command
        on_update (callable, optional): called each iteration as
            on_update(left_value, right_value) -- e.g. to update display
            sliders, or to check whether an image capture was requested.
            Runs on the control loop thread, so keep it fast and avoid
            anything that blocks (like file I/O).

    Returns:
        None
    """
    global _control_thread, _control_loop_running

    stop_control_loop()

    def control_loop():
        global _control_loop_running
        try:
            while _control_loop_running:
                try:
                    left_val = gamepad_state.left
                    right_val = gamepad_state.right

                    if on_update is not None:
                        on_update(left_val, right_val)

                    left_duty, left_mode = stick_to_motor_command(left_val, max_speed)
                    right_duty, right_mode = stick_to_motor_command(right_val, max_speed)

                    rvr.raw_motors(
                        left_mode=left_mode,
                        left_duty_cycle=left_duty,
                        right_mode=right_mode,
                        right_duty_cycle=right_duty,
                    )
                except Exception:
                    # Print any error immediately -- exceptions raised on a
                    # background thread inside Jupyter can otherwise
                    # disappear without a trace.
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
                print('Safety net: sent explicit zero-speed command on control loop exit.')
            except Exception:
                traceback.print_exc()

    _control_loop_running = True
    _control_thread = threading.Thread(name='control_loop_thread', target=control_loop, daemon=True)
    _control_thread.start()
    print('Control loop started -- drive with the gamepad now.')


def stop_control_loop():
    """Stops the control loop started by start_control_loop(), if one is
    running. Safe to call even if no loop is running.

    Returns:
        None
    """
    global _control_loop_running, _control_thread

    if _control_thread is not None and _control_thread.is_alive():
        _control_loop_running = False
        _control_thread.join(timeout=1)
        print('Control loop stopped.')

    _control_thread = None
