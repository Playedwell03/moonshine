#!/usr/bin/env python3
import argparse
import time

from moonshine_voice import MicTranscriber, TranscriptEventListener, get_model_for_language


class ConsoleListener(TranscriptEventListener):
    def __init__(
        self,
        final_only=False,
        merge_window=0.0,
        silence_seconds=0.0,
        min_chars=0,
        drop_question=False,
    ):
        self.final_only = final_only
        self.merge_window = merge_window
        self.silence_seconds = silence_seconds
        self.min_chars = min_chars
        self.drop_question = drop_question
        self._last_complete_ts = None
        self._current_line = ""
        self._last_print_len = 0
        self._pending_line = ""
        self._pending_updated = False

    def on_line_started(self, event):
        if self.final_only:
            return
        if getattr(event, "line", None):
            print(f"\n> {event.line.text}", flush=True)

    def on_line_text_changed(self, event):
        if self.final_only:
            return
        if getattr(event, "line", None):
            print(f"\r> {event.line.text}", end="", flush=True)

    def on_line_completed(self, event):
        if getattr(event, "line", None):
            if self.final_only:
                text = (event.line.text or "").strip()
                if self.drop_question and text.endswith("?"):
                    return
                if self.min_chars and len(text) < self.min_chars:
                    return
            if self.merge_window and self.final_only:
                now = time.monotonic()
                if self._last_complete_ts is None or (now - self._last_complete_ts) > self.merge_window:
                    if self._last_complete_ts is not None:
                        if not self.silence_seconds:
                            print("", flush=True)
                    self._current_line = event.line.text
                else:
                    self._current_line = (self._current_line + " " + event.line.text).strip()
                if self.silence_seconds:
                    self._pending_line = self._current_line
                    self._pending_updated = True
                else:
                    line_out = f"> {self._current_line}"
                    padding = " " * max(0, self._last_print_len - len(line_out))
                    print(f"\r{line_out}{padding}", end="", flush=True)
                    self._last_print_len = len(line_out)
                self._last_complete_ts = now
            else:
                if self.silence_seconds and self.final_only:
                    self._pending_line = event.line.text
                    self._pending_updated = True
                    self._last_complete_ts = time.monotonic()
                else:
                    print(f"\r> {event.line.text}", flush=True)

    def flush_if_idle(self):
        if not self.silence_seconds or not self._pending_updated:
            return
        if self._last_complete_ts is None:
            return
        now = time.monotonic()
        if (now - self._last_complete_ts) >= self.silence_seconds:
            print(f"\r> {self._pending_line}", flush=True)
            self._pending_line = ""
            self._pending_updated = False
            self._last_print_len = 0


def main():
    parser = argparse.ArgumentParser(description="Realtime mic transcription with Moonshine.")
    parser.add_argument("--language", default="ko", help="Language code (default: ko).")
    parser.add_argument(
        "--update-interval",
        type=float,
        default=0.5,
        help="Seconds between streaming updates (default: 0.5).",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=1024,
        help="Audio frames per block (default: 1024).",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Sample rate (default: 16000).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of channels (default: 1).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Sounddevice input device index (default: system default).",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Only print completed lines (avoid partial '?').",
    )
    parser.add_argument(
        "--merge-window",
        type=float,
        default=0.0,
        help="Merge completed lines within N seconds (final-only).",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=0.0,
        help="Only emit after N seconds of no new completions.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=0,
        help="Drop completed lines shorter than N chars.",
    )
    parser.add_argument(
        "--drop-question",
        action="store_true",
        help="Drop completed lines that end with '?'.",
    )
    args = parser.parse_args()

    model_path, model_arch = get_model_for_language(args.language)
    transcriber = MicTranscriber(
        model_path=model_path,
        model_arch=model_arch,
        update_interval=args.update_interval,
        device=args.device,
        samplerate=args.samplerate,
        channels=args.channels,
        blocksize=args.blocksize,
    )
    listener = ConsoleListener(
        final_only=args.final_only,
        merge_window=args.merge_window,
        silence_seconds=args.silence_seconds,
        min_chars=args.min_chars,
        drop_question=args.drop_question,
    )
    transcriber.add_listener(listener)

    transcriber.start()
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
            listener.flush_if_idle()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            transcriber.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
