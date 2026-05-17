#!/usr/bin/env python3
import argparse
import time

from moonshine_voice import MicTranscriber, TranscriptEventListener, get_model_for_language


DEFAULT_COMMAND_MAP = {
    "창문 열어줘": [
        "창문열어줘",
        "창문 열어 줘",
        "창문여러줘",
        "창문 열어조",
        "참문 열어줘",
        "상문 봐",
        "창문 봐",
    ],
    "창문 닫아줘": [
        "창문닫아줘",
        "창문 닫아 줘",
        "창문 다다줘",
        "창문 닫아조",
    ],
}


def _normalize(s):
    return "".join(ch for ch in s.lower().strip() if not ch.isspace() and ch not in "?.!,~")


def _build_map(default_enabled, user_mappings):
    canonical_to_aliases = {}
    if default_enabled:
        canonical_to_aliases.update(DEFAULT_COMMAND_MAP)
    for raw in user_mappings:
        if "=" not in raw:
            continue
        canon, aliases = raw.split("=", 1)
        canon = canon.strip()
        alias_list = [x.strip() for x in aliases.split(",") if x.strip()]
        if canon:
            canonical_to_aliases.setdefault(canon, []).extend(alias_list)

    alias_to_canon = {}
    for canon, aliases in canonical_to_aliases.items():
        alias_to_canon[_normalize(canon)] = canon
        for alias in aliases:
            alias_to_canon[_normalize(alias)] = canon
    return alias_to_canon


class ConsoleListener(TranscriptEventListener):
    def __init__(
        self,
        final_only=False,
        merge_window=0.0,
        silence_seconds=0.0,
        min_chars=0,
        drop_question=False,
        alias_to_canon=None,
    ):
        self.final_only = final_only
        self.merge_window = merge_window
        self.silence_seconds = silence_seconds
        self.min_chars = min_chars
        self.drop_question = drop_question
        self.alias_to_canon = alias_to_canon or {}
        self._last_complete_ts = None
        self._current_line = ""
        self._last_print_len = 0
        self._pending_line = ""
        self._pending_updated = False

    def _map_text(self, text):
        key = _normalize(text)
        return self.alias_to_canon.get(key, text)

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
            mapped = self._map_text((event.line.text or "").strip())
            if self.final_only:
                if self.drop_question and mapped.endswith("?"):
                    return
                if self.min_chars and len(mapped) < self.min_chars:
                    return
            if self.merge_window and self.final_only:
                now = time.monotonic()
                if self._last_complete_ts is None or (now - self._last_complete_ts) > self.merge_window:
                    if self._last_complete_ts is not None:
                        if not self.silence_seconds:
                            print("", flush=True)
                    self._current_line = mapped
                else:
                    self._current_line = (self._current_line + " " + mapped).strip()
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
                    self._pending_line = mapped
                    self._pending_updated = True
                    self._last_complete_ts = time.monotonic()
                else:
                    print(f"\r> {mapped}", flush=True)

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
    parser = argparse.ArgumentParser(description="Realtime mic transcription with Moonshine (+ command mapping).")
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
    parser.add_argument(
        "--no-default-command-map",
        action="store_true",
        help="Disable built-in Korean command correction map.",
    )
    parser.add_argument(
        "--command-map",
        action="append",
        default=[],
        metavar="CANON=ALIAS1,ALIAS2",
        help="Add command mapping rules. Can be repeated.",
    )
    args = parser.parse_args()

    alias_to_canon = _build_map(not args.no_default_command_map, args.command_map)

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
        alias_to_canon=alias_to_canon,
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
