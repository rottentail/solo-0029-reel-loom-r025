"""A bounded production walkthrough using generated audiovisual media and HTTP."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import time
from .config import Settings
from .application import Application
from .api import Api
from .http import MediaHttpServer
from .client import Client
from .errors import LoomError, require

WORKFLOWS = ("ingest", "revisions", "looks", "render", "signal", "waveform", "loudness",
             "sheet", "captions", "imprint", "cancel-retry", "recovery", "resources", "matrix", "conformance", "lineage")


def expect_error(action, code):
    try:
        action()
    except LoomError as exc:
        require(exc.code == code, "walkthrough_error", "Unexpected workflow rejection", expected=code, actual=exc.code)
        return exc.as_dict()
    raise RuntimeError("The walkthrough expected an operation to be rejected: " + code)


class Walkthrough:
    def __init__(self, settings):
        self.settings = settings
        self.app = Application(settings, start_workers=False)
        self.server = MediaHttpServer(("127.0.0.1", 0), Api(self.app))
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        self.thread.start()
        self.client = Client(f"http://127.0.0.1:{self.server.server_port}/loom/v1")

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.app.close()

    def setup(self):
        with self.app.blobs.temporary(".mp4") as sample:
            self.app.processes.run(["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=3",
                                    "-f", "lavfi", "-i", "sine=frequency=659:sample_rate=48000:duration=3",
                                    "-af", "volume=enable='between(t,1,1.5)':volume=0",
                                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(sample)])
            self.content = sample.read_bytes()
        self.asset = self.client.ingest(self.content, "harbor-tone")
        self.slices = [{"asset": self.asset["id"], "in": .2, "out": 1.2, "label": "开场"},
                       {"asset": self.asset["id"], "in": 1.6, "out": 2.8, "label": "收束"}]
        self.reel = self.client.create_reel("回声样片", self.slices, "demo-reel")
        self.look = self.client.put_look("preview", {"kind": "video", "width": 320, "height": 180})
        self.audio_look = self.client.put_look("listening", {"kind": "audio", "sample_rate": 48000})
        self.ticket = self.client.render(self.reel["id"], "preview", key="demo-render")
        return self

    def finish(self):
        ticket = self.client.request("GET", "/tickets/"+self.ticket["id"])
        if ticket["state"] == "queued":
            self.app.renderer.run_once()
        ticket = self.client.wait(self.ticket["id"], timeout=5)
        require(ticket["state"] == "succeeded", "walkthrough_render", "Walkthrough render did not succeed", ticket=ticket)
        return ticket

    def run(self, name):
        c, a = self.client, self.asset["id"]
        if name == "ingest":
            duplicate = c.ingest(self.content, "duplicate-name")
            require(duplicate["id"] == a, "walkthrough_dedup", "Identical ready media should be reused")
            failure = expect_error(lambda: c.ingest(b"not-media", "bad"), "unreadable_media")
            return {"asset": self.asset, "deduplicated": True, "failure": failure}
        if name == "revisions":
            updated = c.replace_reel(self.reel["id"], 1, "回声样片·修订", list(reversed(self.slices)))
            old = c.request("GET", "/reels/"+self.reel["id"]+"?revision=1")
            require(old["hash"] == self.reel["hash"] and updated["revision"] == 2,
                    "walkthrough_revision", "Historical reel changed")
            failure = expect_error(lambda: c.replace_reel(self.reel["id"], 1, "旧稿", self.slices), "revision_conflict")
            return {"head": updated["head"], "historical_hash": old["hash"], "failure": failure}
        if name == "looks":
            new = c.put_look("preview", {"kind": "video", "width": 640, "height": 360}, 1)
            ticket = c.request("GET", "/tickets/"+self.ticket["id"])
            require(ticket["look_version"] == 1 and new["version"] == 2, "walkthrough_look", "Queued look drifted")
            failure = expect_error(lambda: c.put_look("preview", {"kind":"video"}, 1), "revision_conflict")
            return {"new_version": 2, "frozen_version": ticket["look_version"], "failure": failure}
        if name == "render":
            ticket = self.finish()
            body = {"reel": self.reel["id"], "look": "preview"}
            replay = c.request("POST", "/tickets", body, {"Idempotency-Key":"demo-render"})
            require(replay["id"] == ticket["id"], "walkthrough_idempotency", "Render replay changed identity")
            failure = expect_error(lambda: c.request("POST", "/tickets", {**body,"look":"listening"},
                                                   {"Idempotency-Key":"demo-render"}), "idempotency_conflict")
            audio = c.render(self.reel["id"], "listening")
            self.app.renderer.run_once()
            audio = c.wait(audio["id"], 5)
            require(audio["state"] == "succeeded", "walkthrough_audio", "Audio rendition did not succeed")
            return {"video": ticket, "audio": audio, "failure": failure}
        if name in ("signal", "waveform", "loudness"):
            result = c.request("POST", f"/assets/{a}/{name}", {})
            replay = c.request("POST", f"/assets/{a}/{name}", {})
            require(result["id"] == replay["id"], "walkthrough_analysis", "Analysis cache should be content/settings stable")
            failure = expect_error(lambda: c.request("POST", f"/assets/{a}/{name}", {"unsupported": 1}), "unknown_fields")
            return {"analysis": result, "cached": True, "failure": failure}
        if name == "sheet":
            result = c.request("POST", f"/assets/{a}/contact-sheet", {"count": 6, "columns": 3})
            media = c.request("GET", "/artifacts/"+result["id"]+"/content", raw=True)
            require(media.startswith(b"\xff\xd8"), "walkthrough_sheet", "Contact sheet is not JPEG")
            failure = expect_error(lambda: c.request("POST", f"/assets/{a}/contact-sheet", {"count":1,"columns":3}), "sheet_layout")
            return {"artifact": result, "bytes": len(media), "failure": failure}
        if name == "captions":
            # Source cues may be split around the cut, overlap each other and be
            # repeated by another track; the projection must read naturally.
            srt = ("1\n00:00:00,000 --> 00:00:02,900\n潮声\n\n"
                   "2\n00:00:00,300 --> 00:00:00,900\n近句\n\n"
                   "3\n00:00:00,800 --> 00:00:01,100\n断句\n\n"
                   "4\n00:00:01,100 --> 00:00:01,300\n断句\n\n"
                   "5\n00:00:01,800 --> 00:00:02,500\n晚句\n")
            track = c.request("POST", f"/assets/{a}/cues", {"language": "zh-CN", "srt": srt})
            duplicate = c.request("POST", f"/assets/{a}/cues",
                                  {"language": "zh-CN", "srt": "1\n00:00:00,000 --> 00:00:02,900\n潮声\n"})
            english = c.request("POST", f"/assets/{a}/cues",
                                {"language": "en", "srt": "1\n00:00:00,300 --> 00:00:00,900\nTide\n"})
            projected = c.request("POST", f'/reels/{self.reel["id"]}/captions',
                                  {"revision": 1, "tracks": [track["id"], duplicate["id"], english["id"]]})
            zh_block = next(block for block in projected["languages"] if block["language"] == "zh-CN")
            en_block = next(block for block in projected["languages"] if block["language"] == "en")
            zh_cues = [cue for cue in projected["cues"] if cue["language"] == "zh-CN"]
            en_cues = [cue for cue in projected["cues"] if cue["language"] == "en"]
            by_text = {cue["text"]: cue for cue in zh_cues}
            # A cue spanning the cut projects through both slices but reads once.
            tide = by_text["潮声"]
            require(len(zh_cues) == 4 and len({c["text"] for c in zh_cues}) == 4,
                    "walkthrough_captions", "Cut-up and repeated cues must merge into one reading cue each")
            require(tide["tracks"] == sorted([track["id"], duplicate["id"]])
                    and len(tide["origins"]) == 4
                    and sorted({o["slice"] for o in tide["origins"]}) == [0, 1],
                    "walkthrough_captions", "Merged cue lost its source-track or slice provenance")
            # Source segments split around the cut join back into one cue.
            require(len(by_text["断句"]["origins"]) == 2, "walkthrough_captions",
                    "Split source cues around a cut should continue as one cue")
            # Different text is never joined: later dialogue cannot appear early.
            require(by_text["晚句"]["start_tick"] == 1_200_000 and by_text["近句"]["start_tick"] == 100_000,
                    "walkthrough_captions", "Merging pulled later dialogue onto an earlier interval")
            # Languages are processed independently and get their own SRT member.
            require(en_block["cue_count"] == 1 and en_block["srt_file"] == "captions.en.srt"
                    and zh_block["srt_file"] == "captions.zh-CN.srt", "walkthrough_captions",
                    "Languages must project independently with per-language SRT files")
            require(all(cue["language"] == "en" for cue in en_cues), "walkthrough_captions",
                    "Language blocks must not mix")
            # Reusing the same picture twice must not flash the same line twice.
            repeated = c.create_reel("回声复用", [
                {"asset": a, "in": .3, "out": .8, "label": "第一次"},
                {"asset": a, "in": .3, "out": .8, "label": "第二次"}])
            replay = c.request("POST", f'/reels/{repeated["id"]}/captions',
                               {"revision": 1, "tracks": [english["id"]]})
            only = replay["cues"][0]
            require(only["text"] == "Tide" and len(only["origins"]) == 2
                    and only["start_tick"] == 0 and only["end_tick"] == 1_000_000
                    and len({o["slice_id"] for o in only["origins"]}) == 2,
                    "walkthrough_captions", "Repeated picture should read as one continuous cue with both slices provenance")
            failure = expect_error(lambda: c.request("POST", f"/assets/{a}/cues", {"language": "zh", "srt": "broken"}), "cue_format")
            return {"projection": projected, "track": track, "repeated": replay, "failure": failure}
        if name == "imprint":
            zh_track = c.request("POST", f"/assets/{a}/cues",
                                 {"language": "zh-CN", "srt": "1\n00:00:00,000 --> 00:00:02,900\n潮声\n"})
            en_track = c.request("POST", f"/assets/{a}/cues",
                                 {"language": "en", "srt": "1\n00:00:00,000 --> 00:00:02,900\nTide\n"})
            failure = expect_error(lambda: c.imprint(self.ticket["id"], [zh_track["id"]]), "imprint_state")
            ticket = self.finish()
            imprint = c.imprint(ticket["id"], [zh_track["id"]])
            frozen = imprint["manifest"]["caption"]
            require([f["name"] for f in frozen["files"]] == ["captions.zh-CN.srt"],
                    "walkthrough_imprint", "First delivery must freeze only the selected language")
            require(frozen["tracks"][0]["id"] == zh_track["id"], "walkthrough_imprint",
                    "Frozen package must record the source tracks it was built from")
            verified = c.request("GET", f'/imprints/{imprint["id"]}/verify')
            require(set(verified["verified_files"]) >= {"captions.zh-CN.srt"}, "walkthrough_imprint",
                    "Frozen caption members must be verified from the package")
            self.client.replace_reel(self.reel["id"], 1, "新的草剪", list(reversed(self.slices)))
            unchanged = c.imprint(ticket["id"], [zh_track["id"]])
            require(imprint["id"] == unchanged["id"], "walkthrough_imprint", "Upstream editing changed frozen imprint")
            # A later delivery may add a new language track, but it must never be
            # back-filled into the already-frozen imprint.
            revised = c.imprint(ticket["id"], [zh_track["id"], en_track["id"]])
            require(revised["id"] != imprint["id"], "walkthrough_imprint",
                    "Adding a language track must produce a distinguishable new delivery")
            fresh = c.request("GET", f'/imprints/{revised["id"]}/verify')
            require(set(fresh["verified_files"]) >= {"captions.zh-CN.srt", "captions.en.srt"},
                    "walkthrough_imprint", "New delivery must verify every language member")
            old = c.request("GET", f'/imprints/{imprint["id"]}/verify')
            require(old["valid"] and [f["name"] for f in imprint["manifest"]["caption"]["files"]] == ["captions.zh-CN.srt"],
                    "walkthrough_imprint", "New tracks must not be added to a historical delivery")
            return {"imprint": imprint, "revised": revised, "verified": verified, "failure": failure}
        if name == "cancel-retry":
            cancelled = c.request("POST", f'/tickets/{self.ticket["id"]}/cancel')
            require(cancelled["state"] == "cancelled", "walkthrough_cancel", "Queued render did not cancel")
            retried = c.request("POST", f'/tickets/{self.ticket["id"]}/retry')
            ticket = self.finish()
            failure = expect_error(lambda: c.request("POST", f'/tickets/{self.ticket["id"]}/retry'), "retry_state")
            require(retried["attempt"] == 2, "walkthrough_attempt", "Retry attempt was not incremented")
            return {"ticket": ticket, "failure": failure}
        if name == "recovery":
            claimed = self.app.tickets.claim()
            self.close()
            with Application(self.settings, start_workers=False) as restarted:
                recovered = restarted.tickets.get(claimed["id"])
                require(recovered["state"] == "failed" and recovered["error"]["error"] == "worker_interrupted",
                        "walkthrough_recovery", "Interrupted render was not recovered")
                restarted.tickets.retry(claimed["id"])
                finished = restarted.renderer.run_once()
                require(finished["state"] == "succeeded", "walkthrough_recovery", "Recovered render could not retry")
                return {"interrupted": recovered, "retried_state": finished["state"]}
        if name == "resources":
            c.request("POST", f"/assets/{a}/retire")
            failure = expect_error(lambda: c.create_reel("退役来源", self.slices), "asset_retired")
            ticket = self.finish()
            sweep = c.request("POST", "/resources/sweep", {"dry_run":False})
            status = c.request("GET", "/resources")
            require(status["consistent"] and not sweep["removed"], "walkthrough_resources", "Referenced source was collected")
            return {"reconcile":status,"sweep":sweep,"render_state":ticket["state"],"failure":failure}
        if name == "matrix":
            body = {"reel":self.reel["id"],"looks":[{"id":"preview"},{"id":"listening"}]}
            matrix = c.request("POST", "/matrices", body, {"Idempotency-Key":"demo-matrix"})
            before = len(c.request("GET", "/tickets")["items"])
            failure = expect_error(lambda: c.request("POST", "/matrices", {**body,"looks":[{"id":"preview"},{"id":"missing-look"}]}), "not_found")
            require(len(c.request("GET", "/tickets")["items"]) == before, "walkthrough_matrix", "Invalid matrix committed partial work")
            while self.app.renderer.run_once() is not None:
                pass
            finished = c.request("GET", "/matrices/"+matrix["id"])
            require(finished["state"] == "succeeded", "walkthrough_matrix", "Matrix did not complete")
            return {"matrix": finished,"failure":failure}
        if name == "conformance":
            ticket = self.finish()
            artifact = ticket["artifacts"][0]
            result = c.request("GET", f'/artifacts/{artifact["id"]}/conformance')
            require(result["valid"], "walkthrough_conformance", "Rendered media violates frozen look", report=result)
            failure = expect_error(lambda: c.request("GET", "/artifacts/missing/conformance"), "not_found")
            return {"conformance":result,"failure":failure}
        if name == "lineage":
            self.finish()
            imprint = c.imprint(self.ticket["id"])
            result = c.request("GET", f'/tickets/{self.ticket["id"]}/lineage')
            require(any(n["id"] == imprint["id"] for n in result["nodes"]), "walkthrough_lineage", "Imprint missing from lineage")
            failure = expect_error(lambda: c.request("GET", "/tickets/missing/lineage"), "not_found")
            return {"lineage":result,"failure":failure}
        raise ValueError("Unknown walkthrough: " + name)


def main():
    parser = argparse.ArgumentParser(description="Run a bounded audiovisual production walkthrough")
    parser.add_argument("--workflow", choices=WORKFLOWS, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reel-loom-demo-") as folder:
        settings = replace(Settings.from_env(), root=Path(folder), render_timeout=30)
        demo = Walkthrough(settings)
        try:
            demo.setup()
            result = demo.run(args.workflow)
            print(json.dumps({"workflow":args.workflow,"ok":True,"result":result}, ensure_ascii=False))
        finally:
            if args.workflow != "recovery":
                demo.close()


if __name__ == "__main__":
    main()
