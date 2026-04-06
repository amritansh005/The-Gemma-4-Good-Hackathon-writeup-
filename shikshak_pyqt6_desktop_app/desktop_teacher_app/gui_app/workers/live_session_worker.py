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
# Why this exists:
# The old version monkey-patched builtins.print with a bound instance method:
#     builtins.print = self._capture_print
# That can break some code paths which expect the capture callable to exist
# at module scope, causing warnings like:
#     module 'gui_app.workers.live_session_worker' has no attribute '_capture_print'
#
# To preserve all current behavior without changing functionality, we route
# print() through a module-level dispatcher and let the active PrintCapture
# instance handle the text.
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
                # Don't emit "Speaking" here — the event loop's TTS
                # state tracking handles Speaking/Listening transitions
                # based on _tts_is_playing().  Emitting "Speaking" here
                # can race with the TTS-stopped detection and leave the
                # UI stuck on "Speaking" after TTS finishes.
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
            # The event loop NEVER blocks. finalize_turn runs in a
            # background thread (Option A) so VAD events keep flowing,
            # enabling real-time interruption detection at all times.

            # Track whether we already emitted "Listening" for the
            # current interruption so we don't spam the signal on
            # every frame.
            _interruption_listening_emitted = False

            # Track TTS playback state to emit Speaking/Listening
            # transitions directly from the event loop.
            _last_tts_playing = False

            # Active PrintCapture that must outlive finalize_turn
            # because _bg_stream_tts_and_memory (spawned inside
            # finalize_turn) prints Teacher+/Teacher: lines AFTER
            # finalize_turn returns.  We keep it alive here and
            # release it when TTS finishes.
            _active_capture: Optional[PrintCapture] = None

            try:
                for event in streamer.stream_events():
                    if self._stop_requested:
                        break

                    # ── TTS state tracking ──
                    # Check every frame whether TTS started/stopped and
                    # update the GUI status accordingly.
                    _tts_now = vcc._tts_is_playing()
                    if _tts_now and not _last_tts_playing:
                        # TTS just started playing
                        self.status_changed.emit("Speaking")
                    elif not _tts_now and _last_tts_playing:
                        # TTS just stopped playing.
                        # The "Teacher: ..." final print happens AFTER
                        # stream_and_play sets is_playing=False (it prints
                        # after producer.join).  So we cannot release the
                        # capture here — we'd miss the final teacher text.
                        # Instead, just update the status.  The capture
                        # will be released below when saw_final_teacher
                        # becomes True.

                        # Revert to Listening unless an interruption is
                        # being processed (user started speaking during
                        # the TTS → new turn is being captured).
                        with state.lock:
                            is_finalizing = state.finalizing
                            is_interrupted = state.interruption.confirmed
                        if not is_finalizing and not is_interrupted:
                            self.status_changed.emit("Listening")
                    _last_tts_playing = _tts_now

                    # ── Deferred PrintCapture cleanup ──
                    # Release the capture when:
                    # (a) The bg TTS thread printed "Teacher: ..." (final
                    #     text is in the chat), OR
                    # (b) cap.done was set (filtered/hallucinated, no bg
                    #     thread spawned), OR
                    # (c) TTS stopped and the capture saw a teacher reply
                    #     but the final "Teacher:" line hasn't arrived yet
                    #     — give it a grace period by checking
                    #     saw_final_teacher.
                    if _active_capture is not None:
                        cap = _active_capture
                        should_release = False

                        if cap.done:
                            # Filtered / hallucinated / no bg thread
                            should_release = True
                        elif cap.saw_final_teacher:
                            # "Teacher: ..." was printed — we have the
                            # full answer in the chat history now.
                            should_release = True

                        if should_release:
                            _active_capture = None
                            self.live_teacher_text.emit("")
                            cap.release()
                            # The full LLM+TTS cycle is done.  Ensure
                            # we're in Listening state (the TTS-stopped
                            # transition may have already emitted it, but
                            # "Teacher:" prints after TTS stops, so emit
                            # again to be safe).
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
                            # Don't reset the streamer if a new turn is
                            # already in progress — the user started
                            # speaking before/during finalization and we
                            # must not kill their ongoing speech capture.
                            if state.turn.speech_active:
                                # Just clear the flag — the speech frames
                                # are already being collected into the
                                # current turn.
                                state.needs_streamer_reset = False
                                logger.info(
                                    "Skipped streamer reset — speech already active"
                                )
                            else:
                                state.needs_streamer_reset = False
                                streamer.reset()
                            # After a reset, clear the flag so the next
                            # interruption can trigger "Listening" again.
                            _interruption_listening_emitted = False

                    # ── Replay buffered speech ──
                    # If speech was captured during finalization, replay
                    # the buffered frames into a fresh turn now that
                    # finalization is complete.
                    with state.lock:
                        has_buffer = (
                            not state.finalizing
                            and state.pending_speech_active
                            and len(state.pending_speech_buffer) > 0
                        )
                    if has_buffer:
                        with state.lock:
                            buffered = list(state.pending_speech_buffer)
                            state.pending_speech_buffer.clear()
                            state.pending_speech_active = False
                        # Start a new turn with the buffered frames
                        logger.info(
                            "Replaying %d buffered speech frames from during finalization",
                            len(buffered),
                        )
                        with state.lock:
                            if not state.turn.speech_active and not state.finalizing:
                                turn_manager.start_turn(state.turn)
                                if not vcc._tts_is_playing():
                                    self.status_changed.emit("Listening")
                                self.note_changed.emit("Student speech detected")
                        replay_finalized = False
                        for buf_frame in buffered:
                            with state.lock:
                                if not state.turn.speech_active or state.finalizing:
                                    break
                                turn_manager.append_frame(
                                    state.turn,
                                    buf_frame["pcm_bytes"],
                                    is_speech=buf_frame["is_speech"],
                                )
                                state.is_speech_flags.append(buf_frame["is_speech"])
                                decision = turn_manager.evaluate(state.turn)

                            if decision.action == "finalize":
                                logger.info(
                                    "Buffered speech finalized during replay | reason=%s | frames=%d",
                                    decision.reason, len(buffered),
                                )
                                self.status_changed.emit("Thinking")

                                # Release any stale capture
                                if _active_capture is not None:
                                    _active_capture.release()
                                    _active_capture = None

                                capture = PrintCapture(self)
                                capture.__enter__()
                                _active_capture = capture

                                def _finalize_replay_bg(cap=capture):
                                    try:
                                        vcc.finalize_turn(
                                            state, stt, turn_manager, session_id,
                                            streamer, ser_model, speaker_verifier,
                                        )
                                    except Exception as exc:
                                        logger.exception("finalize_turn (replay) failed")
                                        self.error_occurred.emit(str(exc))
                                    finally:
                                        self.live_student_text.emit("")
                                        if not cap.saw_valid_student and not cap.saw_tts_resume:
                                            cap.done = True
                                            if not self._stop_requested:
                                                self.status_changed.emit("Listening")

                                threading.Thread(
                                    target=_finalize_replay_bg,
                                    daemon=True,
                                    name="finalize-replay",
                                ).start()
                                replay_finalized = True
                                break
                            elif decision.action == "discard":
                                logger.info(
                                    "Buffered speech discarded during replay | reason=%s",
                                    decision.reason,
                                )
                                state.reset()
                                replay_finalized = True
                                break

                        if not replay_finalized:
                            vcc.maybe_launch_partial_transcription(state, stt, turn_manager)

                    # Interruption handling — runs on EVERY speech_frame,
                    # even while finalize_turn is running in background
                    if event.event_type == "speech_frame":
                        vcc.maybe_handle_tts_interruption(
                            state=state,
                            event=event,
                            stt=stt,
                            noise_gate=streamer.noise_gate,
                        )

                        # Check if the interruption was just confirmed
                        # by the background ASR thread. The print
                        # "[Assistant interrupted by user]" happens on
                        # that thread and PrintCapture may not see it,
                        # so we detect it directly from state.
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
                            if state.finalizing:
                                if vcc._tts_is_playing():
                                    # TTS is active → potential interruption.
                                    # Buffer for the interruption pipeline.
                                    if not state.pending_speech_active:
                                        state.pending_speech_active = True
                                        state.pending_speech_buffer.clear()
                                        logger.info("Buffering speech during finalization (TTS playing)")
                                else:
                                    # TTS has NOT started → user is speaking
                                    # during the Thinking phase.  Cancel the
                                    # old response and start fresh.
                                    # Preserve buffered frames so early words
                                    # aren't lost.
                                    saved_frames = list(state.pending_speech_buffer)
                                    logger.info(
                                        "New speech during Thinking phase — cancelling pending response | buffered_frames=%d",
                                        len(saved_frames),
                                    )
                                    vcc.cancel_pending_response()
                                    state.finalizing = False
                                    state.pending_speech_buffer.clear()
                                    state.pending_speech_active = False
                                    state.reset()
                                    turn_manager.start_turn(state.turn)
                                    # Replay saved frames into the fresh turn
                                    for buf in saved_frames:
                                        turn_manager.append_frame(
                                            state.turn,
                                            buf["pcm_bytes"],
                                            is_speech=buf["is_speech"],
                                        )
                                        state.is_speech_flags.append(buf["is_speech"])
                                    self.status_changed.emit("Listening")
                                    self.note_changed.emit("Student speech detected (cancelled previous)")
                                    # Release stale PrintCapture
                                    if _active_capture is not None:
                                        _active_capture.release()
                                        _active_capture = None
                                        self.live_teacher_text.emit("")
                            elif not state.turn.speech_active:
                                turn_manager.start_turn(state.turn)
                                if not vcc._tts_is_playing():
                                    self.status_changed.emit("Listening")
                                self.note_changed.emit("Student speech detected")
                        continue

                    if event.event_type == "speech_frame" and event.pcm_bytes:
                        with state.lock:
                            if state.finalizing:
                                if vcc._tts_is_playing():
                                    # TTS active → buffer for interruption pipeline
                                    if state.pending_speech_active:
                                        state.pending_speech_buffer.append({
                                            "pcm_bytes": event.pcm_bytes,
                                            "is_speech": event.is_speech,
                                        })
                                    continue
                                else:
                                    # TTS not playing → cancel old response,
                                    # process frame normally.
                                    # Preserve buffered frames so early words
                                    # aren't lost.
                                    if not state.turn.speech_active:
                                        saved_frames = list(state.pending_speech_buffer)
                                        logger.info(
                                            "Speech frame during Thinking phase — cancelling pending response | buffered_frames=%d",
                                            len(saved_frames),
                                        )
                                        vcc.cancel_pending_response()
                                        state.finalizing = False
                                        state.pending_speech_buffer.clear()
                                        state.pending_speech_active = False
                                        state.reset()
                                        # Release stale PrintCapture
                                        if _active_capture is not None:
                                            _active_capture.release()
                                            _active_capture = None
                                            self.live_teacher_text.emit("")
                                        if event.is_speech:
                                            turn_manager.start_turn(state.turn)
                                            # Replay saved frames
                                            for buf in saved_frames:
                                                turn_manager.append_frame(
                                                    state.turn,
                                                    buf["pcm_bytes"],
                                                    is_speech=buf["is_speech"],
                                                )
                                                state.is_speech_flags.append(buf["is_speech"])
                                            self.status_changed.emit("Listening")
                                            self.note_changed.emit("Student speech detected (cancelled previous)")
                                        else:
                                            continue
                            if not state.turn.speech_active:
                                # If VAD says speech is happening but turn
                                # isn't active, auto-start a turn.  This
                                # handles the race where state.reset()
                                # clears speech_active between speech_start
                                # and the next speech_frame.
                                if event.is_speech:
                                    turn_manager.start_turn(state.turn)
                                    if not vcc._tts_is_playing():
                                        self.status_changed.emit("Listening")
                                    self.note_changed.emit("Student speech detected")
                                else:
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

                            # ── Option A: run finalize_turn in background ──
                            # PrintCapture intercepts print() from BOTH the
                            # finalize_turn thread AND the _bg_stream_tts_and_memory
                            # thread spawned inside finalize_turn.
                            #
                            # IMPORTANT: finalize_turn() spawns
                            # _bg_stream_tts_and_memory as a daemon thread
                            # and returns immediately.  Teacher+/Teacher:
                            # prints happen on THAT bg thread, so we must
                            # NOT exit PrintCapture when finalize_turn
                            # returns.  Instead, we keep a reference in
                            # _active_capture and release it from the event
                            # loop when TTS stops playing.

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
                                    # If valid student text was seen, finalize_turn
                                    # spawned _bg_stream_tts_and_memory which will
                                    # print Teacher+/Teacher: lines.  Do NOT release
                                    # the capture — let the event loop release it
                                    # when TTS stops playing.
                                    #
                                    # If no valid student text (filtered, hallucinated,
                                    # too short, etc.) and no TTS resume, then no bg
                                    # thread was spawned.  Release immediately.
                                    if (
                                        not cap.saw_valid_student
                                        and not cap.saw_tts_resume
                                    ):
                                        cap.done = True  # signal event loop
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
                                # Resume TTS on a background thread so the
                                # main event loop keeps processing VAD events
                                # — the user must be able to interrupt the
                                # resumed playback.
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
                # Release any lingering PrintCapture
                if _active_capture is not None:
                    _active_capture.release()
                    _active_capture = None
                # Restore monkey-patch
                vcc.print_live = original_print_live

        except Exception as exc:
            logger.exception("Desktop live session failed")
            self.error_occurred.emit(str(exc))
        finally:
            self.status_changed.emit("Stopped")
            self.finished.emit()