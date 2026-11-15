import json
import re
from .errors import require, missing
from .values import identifier, now, canonical, text, digest
from .timebase import intersect, stamp, TICKS_PER_SECOND

TIME = re.compile(r"^(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})$")
SAFE_LANGUAGE = re.compile(r"[^A-Za-z0-9._-]+")


def parse_stamp(value):
    match = TIME.fullmatch(value.strip())
    require(match is not None, "cue_time", "Invalid subtitle timestamp")
    h, m, s, ms = map(int, match.groups())
    require(m < 60 and s < 60, "cue_time", "Subtitle minute or second is invalid")
    return ((h*60+m)*60+s)*TICKS_PER_SECOND + ms*1000


def parse_srt(body, duration):
    require(isinstance(body, str) and len(body.encode()) <= 2_000_000,
            "caption_size", "Subtitle input exceeds limit")
    blocks = re.split(r"\n\s*\n", body.lstrip("﻿").replace("\r\n", "\n").strip())
    require(len(blocks) <= 10_000, "cue_limit", "Too many subtitle cues")
    cues = []
    for block in blocks:
        lines = block.splitlines()
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]
        require(len(lines) >= 2 and " --> " in lines[0], "cue_format", "Invalid SRT cue")
        left, right = lines[0].split(" --> ", 1)
        start, end = parse_stamp(left), parse_stamp(right)
        require(0 <= start < end <= duration, "cue_bounds", "Subtitle lies outside source media")
        caption = "\n".join(lines[1:]).strip()
        require(bool(caption) and len(caption) <= 2000, "cue_text", "Invalid subtitle text")
        require(not cues or start >= cues[-1]["start_tick"], "cue_order", "Subtitles must be in start-time order")
        cues.append({"start_tick": start, "end_tick": end, "text": caption})
    return cues


def render_srt(cues):
    return "\n\n".join(f'{i+1}\n{stamp(c["start_tick"])} --> {stamp(c["end_tick"])}\n{c["text"]}'
                       for i, c in enumerate(cues)) + "\n"


def safe_language_name(language, used):
    """Deterministic filesystem-safe SRT member name, unique within one package."""
    base = SAFE_LANGUAGE.sub("_", language).strip("._") or "language"
    name = f"captions.{base}.srt"
    if name in used:
        name = f"captions.{base}-{digest(language)[:8]}.srt"
    used.add(name)
    return name


class Captions:
    def __init__(self, db, assets, reels):
        self.db, self.assets, self.reels = db, assets, reels

    def import_track(self, asset_id, language, body):
        language = text(language, "language", 40)
        with self.db.write_unit() as db:
            asset = self.assets.get(asset_id, db, ready=True)
            cues = parse_srt(body, asset["metadata"]["duration_tick"])
            track_id = identifier("cue")
            db.execute("INSERT INTO cue_tracks VALUES(?,?,?,?,?)",
                       (track_id, asset_id, language, canonical(cues), now()))
            self.db.event(db, asset_id, "cues_imported", {"track": track_id, "count": len(cues)})
            return self.get(track_id, db)

    def get(self, track_id, db=None):
        if db is None:
            with self.db.read() as read:
                return self.get(track_id, read)
        row = db.execute("SELECT * FROM cue_tracks WHERE id=?", (track_id,)).fetchone()
        if row is None:
            raise missing("cue track", track_id)
        return {**dict(row), "cues": json.loads(row["cues"])}

    def _fragments(self, track, reel_document):
        """Project every source cue onto each slice using the same asset.

        Each fragment keeps its full source provenance so merged reading cues
        stay traceable to the immutable source track, cue and slice.
        """
        fragments = []
        for position, item in enumerate(reel_document["slices"]):
            if item["asset"] != track["asset"]:
                continue
            offset = item["at_tick"] - item["in_tick"]
            for cue_index, cue in enumerate(track["cues"]):
                overlap = intersect(cue["start_tick"], cue["end_tick"], item["in_tick"], item["out_tick"])
                if overlap is None:
                    continue
                source_start, source_end = overlap
                fragments.append({
                    "start_tick": source_start + offset,
                    "end_tick": source_end + offset,
                    "text": cue["text"],
                    "track": track["id"],
                    "language": track["language"],
                    "source_cue": cue_index,
                    "source_start_tick": cue["start_tick"],
                    "source_end_tick": cue["end_tick"],
                    "source_text": cue["text"],
                    "slice": position,
                    "slice_id": item["id"],
                    "source_in_tick": item["in_tick"],
                    "source_out_tick": item["out_tick"],
                })
        return fragments

    def _merge_language(self, language, fragments):
        """Turn raw projected fragments into natural reading cues for one language.

        Fragments with identical text are treated as the same line: intervals
        that overlap or touch are unioned into one reading cue, which removes
        flash repeats caused by cut-up source cues, reused picture or
        overlapping tracks even when other dialogue sorts between them.
        Different text is never joined, so later dialogue is never displayed
        early, and genuinely separated occurrences keep a gap between cues.
        Each merged cue retains every source fragment it came from.
        """
        by_text = {}
        for fragment in fragments:
            by_text.setdefault(fragment["text"], []).append(fragment)
        cues = []
        for text_value, same in by_text.items():
            same.sort(key=lambda x: (x["start_tick"], x["end_tick"], x["track"], x["source_cue"], x["slice"]))
            groups, current, current_end = [], None, None
            for fragment in same:
                if current is not None and fragment["start_tick"] <= current_end:
                    current.append(fragment)
                    current_end = max(current_end, fragment["end_tick"])
                else:
                    current = [fragment]
                    groups.append(current)
                    current_end = fragment["end_tick"]
            for group in groups:
                origins, seen = [], set()
                for fragment in group:
                    key = (fragment["track"], fragment["source_cue"], fragment["slice"])
                    if key in seen:
                        continue
                    seen.add(key)
                    origins.append({
                        "track": fragment["track"],
                        "source_cue": fragment["source_cue"],
                        "source_start_tick": fragment["source_start_tick"],
                        "source_end_tick": fragment["source_end_tick"],
                        "source_text": fragment["source_text"],
                        "slice": fragment["slice"],
                        "slice_id": fragment["slice_id"],
                        "source_in_tick": fragment["source_in_tick"],
                        "source_out_tick": fragment["source_out_tick"],
                        "start_tick": fragment["start_tick"],
                        "end_tick": fragment["end_tick"],
                    })
                cues.append({
                    "start_tick": min(f["start_tick"] for f in group),
                    "end_tick": max(f["end_tick"] for f in group),
                    "text": text_value,
                    "language": language,
                    "tracks": sorted({f["track"] for f in group}),
                    "origins": origins,
                })
        cues.sort(key=lambda x: (x["start_tick"], x["end_tick"], x["text"]))
        return cues

    def project(self, reel_id, revision, track_ids):
        require(isinstance(track_ids, list) and 1 <= len(track_ids) <= 64, "track_count", "Expected 1..64 cue tracks")
        require(len(track_ids) == len(set(track_ids)), "duplicate_track", "Cue tracks cannot repeat")
        with self.db.read() as db:
            reel = self.reels.get(reel_id, revision, db)
            tracks = []
            by_language = {}
            for track_id in track_ids:
                track = self.get(track_id, db)
                require(any(item["asset"] == track["asset"] for item in reel["document"]["slices"]),
                        "unrelated_track", "Cue track does not belong to this reel")
                tracks.append(track)
                by_language.setdefault(track["language"], []).extend(self._fragments(track, reel["document"]))
            language_blocks, used_names, srts = [], set(), {}
            for language in sorted(by_language):
                cues = self._merge_language(language, by_language[language])
                member = safe_language_name(language, used_names)
                srts[language] = render_srt(cues)
                language_blocks.append({
                    "language": language,
                    "srt_file": member,
                    "tracks": [t["id"] for t in tracks if t["language"] == language],
                    "cue_count": len(cues),
                    "cues": cues,
                })
            track_details = [{
                "id": track["id"],
                "asset": track["asset"],
                "language": track["language"],
                "cue_count": len(track["cues"]),
                "created": track["created"],
            } for track in tracks]
            projected = [cue for block in language_blocks for cue in block["cues"]]
            projected.sort(key=lambda x: (x["start_tick"], x["language"], x["end_tick"], x["text"]))
            return {
                "reel": reel_id,
                "revision": reel["revision"],
                "reel_hash": reel["hash"],
                "tracks": track_details,
                "languages": [{k: block[k] for k in ("language", "srt_file", "tracks", "cue_count")}
                              for block in language_blocks],
                "cues": projected,
                "srts": srts,
            }
