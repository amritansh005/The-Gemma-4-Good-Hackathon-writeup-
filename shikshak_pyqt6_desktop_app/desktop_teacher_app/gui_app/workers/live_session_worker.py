"""
Thin Qt wrapper around the fully-working backend voice_chat_client.

All heavy lifting (VAD, STT, interruption, TTS control, emotion) is done
by the backend code in STT/voice_chat_client.py.  This worker just:
  1. Runs the backend event loop on a QThread
  2. Emits Qt signals so the GUI can display what's happening

Option A+B: finalize_turn runs in a background thread so the event loop
keeps processing VAD events — enabling real-time interruption even while
the LLM is generating and TTS is speaking.
"""
from __future__ import annotations

import builtins
import logging
import threading
from datetime import datetime
from typing import Optional

import requests
from PyQt6.QtCore import QObject, pyqtSignal

from gui_app.bootstrap import setup_project_imports

setup_project_imports()

from app.config import settings  # type: ignore
from app.services.emotion_service import SERModel  # type: ignore
from app.services.realtime_vad import MicrophoneVADStreamer  # type: ignore
from app.services.stt_service import STTService  # type: ignore
from app.services.turn_manager import TurnManager  # type: ignore

try:
    from app.services.speaker_verification import SpeakerVerificationService  # type: ignore
except Exception:
    SpeakerVerificationService = None

# ── Import the WORKING backend logic directly ──
import voice_chat_client as vcc  # type: ignore

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# SAFE GLOBAL PRINT CAPTURE DISPATCH
# -------------------------------------------------------------------------
_ORIGINAL_PRINT = builtins.print
_ACTIVE_PRINT_CAPTURE: Optional["PrintCapture"] = None
_PRINT_CAPTURE_LOCK = threading.RLock()


def _capture_print(*args, **kwargs) -> None:
    capture: Optional["PrintCapture"]
    with _PRINT_CAPTURE_LOCK:
        capture = _ACTIVE_PRINT_CAPTURE

    if capture is not None:
        capture.handle_print(*args, **kwargs)
    else:
        _ORIGINAL_PRINT(*args, **kwargs)


class PrintCapture:
    """
    Temporarily intercept print() calls from finalize_turn to extract
    student text, teacher text, and emotion for the GUI signals.

    IMPORTANT: finalize_turn() spawns _bg_stream_tts_and_memory as a
    daemon thread and returns immediately.  The Teacher+ / Teacher:
    prints happen on that bg thread AFTER finalize_turn returns.
    Therefore PrintCapture must stay active until the bg TTS thread
    finishes.  We track this with `self.done` — the event loop
    checks it when TTS stops playing and only then calls __exit__.
    """

    def __init__(self, worker: "LiveSessionWorker") -> None:
        self.worker = worker
        self.saw_teacher_reply = False
        self.saw_tts_resume = False
        self.saw_final_teacher = False
        self.saw_valid_student = False  # "You: ..." printed → bg TTS thread spawned
        self.tts_has_stopped = False    # event loop sets when TTS playing→stopped
        self.done = False  # set True when safe to release

    def __enter__(self):
        global _ACTIVE_PRINT_CAPTURE
        with _PRINT_CAPTURE_LOCK:
            _ACTIVE_PRINT_CAPTURE = self
            builtins.print = _capture_print
        return self

    def __exit__(self, *args):
        global _ACTIVE_PRINT_CAPTURE
        with _PRINT_CAPTURE_LOCK:
            if _ACTIVE_PRINT_CAPTURE is self:
                _ACTIVE_PRINT_CAPTURE = None
            builtins.print = _ORIGINAL_PRINT

    def release(self):
        """Mark this capture as done and deactivate it."""
        self.done = True
        self.__exit__(None, None, None)

    def handle_print(self, *args, **kwargs) -> None:
        text = " ".join(str(a) for a in args)

        # Always forward to real print for terminal logging
        _ORIGINAL_PRINT(*args, **kwargs)

        # Extract signals from backend print output
        if text.startswith("You: ") and "[filtered" not in text and "[too short" not in text:
            student_text = text[5:].strip()
            if student_text:
                self.worker.final_student_text.emit(student_text)
                self.worker.live_student_text.emit("")
                self.saw_valid_student = True

        elif text.startswith("Teacher+ "):
            streamed_text = text[9:].strip()
            if streamed_text:
                self.worker.live_teacher_text.emit(streamed_text)
                # Emit "Speaking" to cover the gap between first LLM
                # sentence arriving and TTS actually starting playback.
                # BUT only if TTS hasn't already stopped — otherwise a
                # late Teacher+ would re-set status to "Speaking" after
                # the event loop already switched to "Listening".
                if not self.tts_has_stopped:
                    self.worker.status_changed.emit("Speaking")
                self.saw_teacher_reply = True

        elif text.startswith("Teacher: "):
            teacher_text = text[9:].strip()
            if teacher_text:
                self.worker.final_teacher_text.emit(teacher_text)
                self.worker.live_teacher_text.emit("")
                self.saw_teacher_reply = True
                self.saw_final_teacher = True

        elif text.strip().startswith("[text:"):
            # Emotion line like: [text: neutral | voice: sad (50%) | ...]
            self.worker.emotion_changed.emit(text.strip().strip("[]"))

        elif "[Assistant interrupted" in text:
            self.worker.note_changed.emit("Assistant interrupted by student")
            self.worker.status_changed.emit("Listening")

        elif "Listening..." in text:
            self.worker.status_changed.emit("Listening")
            self.worker.note_changed.emit("Student speech detected")

        elif "[Resuming teacher speech]" in text:
            self.worker.note_changed.emit("Resuming teacher speech")
            self.worker.status_changed.emit("Speaking")
            self.saw_tts_resume = True

        elif "Speaker profile enrolled" in text:
            self.worker.note_changed.emit("Speaker profile enrolled")


class LiveSessionWorker(QObject):
    """Runs the backend voice chat loop and emits GUI signals."""

    status_changed = pyqtSignal(str)
    note_changed = pyqtSignal(str)
    live_student_text = pyqtSignal(str)
    final_student_text = pyqtSignal(str)
    live_teacher_text = pyqtSignal(str)
    final_teacher_text = pyqtSignal(str)
    emotion_changed = pyqtSignal(str)
    session_ready = pyqtSignal(str)
    health_report = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._stop_requested = False
        self._streamer: Optional[MicrophoneVADStreamer] = None

    def stop(self) -> None:
        self._stop_requested = True
        if self._streamer is not None:
            self._streamer.reset()
        # Stop TTS playback
        if vcc._tts_client is not None:
            try:
                vcc._tts_stop_playback()
            except Exception:
                pass

    def run(self) -> None:
        try:
            self.status_changed.emit("Starting")
            self.note_changed.emit("Loading services...")

            # ── Setup ──
            session_id = f"voice-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            # Try to reuse preloaded models
            try:
                from gui_app.preloader import get_preloaded

                models = get_preloaded()
                if models.ready:
                    stt = models.stt or STTService()
                    ser_model = models.ser_model or SERModel()
                    speaker_verifier = models.speaker_verifier
                    self.note_changed.emit("Using preloaded models")
                else:
                    raise RuntimeError("Not preloaded")
            except Exception:
                self.note_changed.emit("Loading STT...")
                stt = STTService()
                self.note_changed.emit("Loading emotion model...")
                ser_model = SERModel()
                speaker_verifier = None
                if SpeakerVerificationService and getattr(settings, "speaker_verification_enabled", False):
                    sv = SpeakerVerificationService(
                        similarity_threshold=settings.speaker_verification_threshold,
                        auto_update_profile=settings.speaker_verification_auto_update,
                        device="cpu",
                    )
                    if sv.is_available:
                        speaker_verifier = sv

            streamer = MicrophoneVADStreamer()
            self._streamer = streamer
            turn_manager = TurnManager()
            state = vcc.LiveTranscriptState()

            # ── Ensure TTS client is initialized ──
            if vcc._tts_client is None:
                try:
                    from tts_client import TTSClient

                    vcc._tts_client = TTSClient(tts_service_url="http://127.0.0.1:5000", timeout=120)
                except Exception as exc:
                    logger.warning("TTSClient not available: %s", exc)

            tts_status = "disabled"
            if vcc._tts_client is not None:
                vcc._tts_client.invalidate_availability_cache()
                if vcc._tts_client.is_available():
                    tts_status = "enabled"
                else:
                    tts_status = "disabled (service not reachable)"
            self.health_report.emit("TTS", tts_status)

            sv_status = "enabled (will enroll on first speech)" if speaker_verifier else "disabled"
            self.health_report.emit("Speaker verification", sv_status)

            # ── Verify teacher backend ──
            try:
                r = requests.get(settings.teacher_chat_url.rsplit("/chat", 1)[0] + "/", timeout=10)
                r.raise_for_status()
                self.note_changed.emit("Teacher backend connected")
            except Exception as exc:
                raise RuntimeError(f"Teacher backend not reachable: {exc}") from exc

            self.session_ready.emit(session_id)
            self.status_changed.emit("Listening")
            self.note_changed.emit("Speak naturally. Live mode is active.")

            # ── Monkey-patch print_live for partial transcription display ──
            original_print_live = vcc.print_live

            def gui_print_live(text: str) -> None:
                clipped = text[:160].replace("You (live): ", "")
                self.live_student_text.emit(clipped)

            vcc.print_live = gui_print_live

            # ── Main event loop ──
            _interruption_listening_emitted = False
            _last_tts_playing = False
            _active_capture: Optional[PrintCapture] = None

            try:
                for event in streamer.stream_events():
                    if self._stop_requested:
                        break

                    # ── TTS state tracking ──
                    _tts_now = vcc._tts_is_playing()
                    if _tts_now and not _last_tts_playing:
                        # TTS just started playing
                        self.status_changed.emit("Speaking")
                    elif not _tts_now and _last_tts_playing:
                        # TTS just stopped playing.
                        # Mark on the capture so Teacher+ won't re-set
                        # status to Speaking after this point.
                        if _active_capture is not None:
                            _active_capture.tts_has_stopped = True

                        # Revert to Listening unless an interruption is
                        # being processed.
                        with state.lock:
                            is_finalizing = state.finalizing
                            is_interrupted = state.interruption.confirmed
                        if not is_finalizing and not is_interrupted:
                            self.status_changed.emit("Listening")
                    _last_tts_playing = _tts_now

                    # ── Deferred PrintCapture cleanup ──
                    if _active_capture is not None:
                        cap = _active_capture
                        should_release = False

                        if cap.done:
                            should_release = True
                        elif cap.saw_final_teacher:
                            should_release = True

                        if should_release:
                            _active_capture = None
                            self.live_teacher_text.emit("")
                            cap.release()
                            # Ensure Listening after full cycle completes
                            if not cap.done and not self._stop_requested:
                                with state.lock:
                                    is_finalizing = state.finalizing
                                    is_interrupted = state.interruption.confirmed
                                if not is_finalizing and not is_interrupted:
                                    self.status_changed.emit("Listening")

                    # finalize_turn (bg thread) cannot safely call
                    # streamer.reset() — do it here on the main thread.
                    with state.lock:
                        if state.needs_streamer_reset:
                            state.needs_streamer_reset = False
                            streamer.reset()
                            _interruption_listening_emitted = False

                    # Interruption handling
                    if event.event_type == "speech_frame":
                        vcc.maybe_handle_tts_interruption(
                            state=state,
                            event=event,
                            stt=stt,
                            noise_gate=streamer.noise_gate,
                        )

                        with state.lock:
                            just_confirmed = (
                                state.interruption.confirmed
                                and not _interruption_listening_emitted
                            )
                        if just_confirmed:
                            _interruption_listening_emitted = True
                            self.status_changed.emit("Listening")
                            self.note_changed.emit("Assistant interrupted by student")

                    if event.event_type == "speech_start":
                        with state.lock:
                            if not state.turn.speech_active and not state.finalizing:
                                turn_manager.start_turn(state.turn)
                                if not vcc._tts_is_playing():
                                    self.status_changed.emit("Listening")
                                self.note_changed.emit("Student speech detected")
                        continue

                    if event.event_type == "speech_frame" and event.pcm_bytes:
                        with state.lock:
                            if not state.turn.speech_active or state.finalizing:
                                continue

                            turn_manager.append_frame(
                                state.turn,
                                event.pcm_bytes,
                                is_speech=event.is_speech,
                            )
                            state.is_speech_flags.append(event.is_speech)
                            decision = turn_manager.evaluate(state.turn)

                        vcc.maybe_launch_partial_transcription(state, stt, turn_manager)

                        if decision.action == "finalize":
                            self.status_changed.emit("Thinking")

                            # Release any stale capture from a previous turn
                            if _active_capture is not None:
                                _active_capture.release()
                                _active_capture = None

                            capture = PrintCapture(self)
                            capture.__enter__()
                            _active_capture = capture

                            def _finalize_bg(cap=capture):
                                try:
                                    vcc.finalize_turn(
                                        state,
                                        stt,
                                        turn_manager,
                                        session_id,
                                        streamer,
                                        ser_model,
                                        speaker_verifier,
                                    )
                                except Exception as exc:
                                    logger.exception("finalize_turn failed")
                                    self.error_occurred.emit(str(exc))
                                finally:
                                    self.live_student_text.emit("")
                                    if (
                                        not cap.saw_valid_student
                                        and not cap.saw_tts_resume
                                    ):
                                        cap.done = True
                                        if not self._stop_requested:
                                            self.status_changed.emit("Listening")

                            threading.Thread(
                                target=_finalize_bg,
                                daemon=True,
                                name="finalize-turn",
                            ).start()

                        elif decision.action == "discard":
                            self.note_changed.emit("Turn ignored: too short")
                            with state.lock:
                                had_interruption = state.interruption.confirmed
                            if had_interruption and vcc._tts_has_resume_audio():
                                self.note_changed.emit("Resuming teacher speech")
                                self.status_changed.emit("Speaking")
                                state.reset()
                                streamer.reset()
                                _interruption_listening_emitted = False

                                def _resume_bg():
                                    try:
                                        vcc._tts_resume_playback()
                                    except Exception:
                                        logger.exception("TTS resume failed")
                                    finally:
                                        if not self._stop_requested:
                                            self.status_changed.emit("Listening")

                                threading.Thread(
                                    target=_resume_bg,
                                    daemon=True,
                                    name="tts-resume",
                                ).start()
                            else:
                                vcc._tts_restore_playback()
                                streamer.reset()
                                state.reset()
                                _interruption_listening_emitted = False
                        else:
                            latest = state.turn.latest_partial()
                            if latest and decision.reason == "soft_pause_resume_window":
                                self.live_student_text.emit(latest + " …")
                        continue

            except Exception as exc:
                logger.exception("Voice chat loop failed")
                self.error_occurred.emit(str(exc))
            finally:
                if _active_capture is not None:
                    _active_capture.release()
                    _active_capture = None
                vcc.print_live = original_print_live

        except Exception as exc:
            logger.exception("Desktop live session failed")
            self.error_occurred.emit(str(exc))
        finally:
            self.status_changed.emit("Stopped")
            self.finished.emit()