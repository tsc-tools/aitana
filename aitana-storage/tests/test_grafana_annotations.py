import json

import pytest

from aitana.storage import grafana_annotations

RU = "volcano:ruapehu"


class FakeResponse:
    def __init__(self, payload=None):
        self.status_code = 200
        self.text = ""
        self.payload = payload

    def json(self):
        return self.payload


class FakeGrafana:
    """Annotations of one organization behind fake GET/PATCH/POST /api/annotations."""

    def __init__(self, annotations):
        self.annotations = {atn["id"]: dict(atn) for atn in annotations}
        self.posted = []
        self.patched = []
        self.headers = []

    def get(self, url, params=None, headers=None):
        assert params == {"type": "annotation", "limit": 100000}
        self.headers.append(headers)
        return FakeResponse([dict(atn) for atn in self.annotations.values()])

    def patch(self, url, json=None, headers=None):
        atn_id = int(url.rsplit("/", 1)[1])
        self.annotations[atn_id].update(json)
        self.patched.append(atn_id)
        self.headers.append(headers)
        return FakeResponse()

    def post(self, url, json=None, headers=None):
        # Grafana reports organization annotations with an empty dashboard and panel 0
        self.headers.append(headers)
        atn_id = max(self.annotations, default=0) + 1
        self.annotations[atn_id] = {"id": atn_id, "login": "sync", "dashboardUID": "",
                                    "panelId": 0, **json}
        self.posted.append(json)
        return FakeResponse({"id": atn_id})


@pytest.fixture
def grafana(monkeypatch):
    def install(annotations):
        fake = FakeGrafana(annotations)
        for method in ["get", "patch", "post"]:
            monkeypatch.setattr(grafana_annotations.requests, method,
                                getattr(fake, method))
        return fake
    return install


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("GRAFANA_USER", raising=False)
    monkeypatch.delenv("GRAFANA_PASSWORD", raising=False)


def annotation(atn_id, text, tags, time=1647158400000, time_end=None, dashboard="rsam",
               panel=2, login="duty"):
    return {"id": atn_id, "dashboardUID": dashboard, "panelId": panel, "time": time,
            "timeEnd": time_end or time, "text": text, "tags": tags, "login": login}


def stored_label(label_id, tags, text="Wind noise\nGusts", time=1647158400000,
                 time_end=1647165600000):
    return {"time": time, "timeEnd": time_end, "title": text.split("\n")[0],
            "description": text, "tags": tags, "id": label_id, "user": "duty",
            "dashboardUID": "rsam", "panelId": 2}


def by_id(labels):
    return {label["id"]: label for label in labels}


def test_sync_union(grafana):
    fake = grafana([
        # known label, retagged in Grafana from noise to tremor
        annotation(1, "Wind noise\nGusts\nId: a1", ["label:tremor", RU],
                   time_end=1647165600000),
        # drawn on another dashboard since the last sync, for a station, no id yet
        annotation(2, "Station fault", ["label:instrument", RU, "series:MAVZ.10.HHZ.rsam"],
                   time=1647244800000, dashboard="overview", panel=7, login="other"),
        # a note, not a label
        annotation(3, "Looked at this", ["note"]),
        # another volcano's label, kept in the same file
        annotation(4, "Steam", ["label:noise", "volcano:whakaari"], time=1647331200000),
        # a label without a volcano tag: kept, but reported
        annotation(5, "Forgot the volcano", ["label:noise"], time=1647417600000),
    ])
    # d1 was deleted in Grafana
    stored = [stored_label("a1", ["label:noise", RU]),
              stored_label("d1", ["label:noise", RU], time=1647072000000)]

    labels, report = grafana_annotations.sync_labels("http://grafana", stored)

    assert (report["restored"], report["added"]) == (1, 3)
    assert [label["title"] for label in report["untagged"]] == ["Forgot the volcano"]
    assert [grafana_annotations._split_id(p["text"])[1] for p in fake.posted] == ["d1"]
    assert "dashboardUID" not in fake.posted[0]
    assert sorted(fake.patched) == [2, 4, 5]

    new_id = grafana_annotations._split_id(fake.annotations[2]["text"])[1]
    assert fake.annotations[2]["text"] == f"Station fault\nId: {new_id}"
    assert [label["time"] for label in labels] == sorted(label["time"] for label in labels)
    assert len(labels) == 5
    labels = by_id(labels)
    assert labels["a1"]["tags"] == ["label:tremor", RU]
    assert labels["a1"]["description"] == "Wind noise\nGusts"
    assert labels["d1"] == stored[1]
    station = labels[new_id]
    assert station["tags"] == ["label:instrument", RU, "series:MAVZ.10.HHZ.rsam"]
    assert station["timeEnd"] is None
    assert (station["user"], station["dashboardUID"], station["panelId"]) == (
        "other", "overview", 7)
    # without GRAFANA_USER and GRAFANA_PASSWORD, credentials can only come from the URL
    assert fake.headers == [{}] * 5


def test_sync_sends_basic_auth(grafana, monkeypatch):
    monkeypatch.setenv("GRAFANA_USER", "sync")
    monkeypatch.setenv("GRAFANA_PASSWORD", "s3cret")
    fake = grafana([annotation(1, "Station fault", ["label:instrument", RU])])
    grafana_annotations.sync_labels("http://grafana",
                                    [stored_label("a1", ["label:noise", RU])])
    # GET, PATCH of the new label's id, POST restoring a1; base64 of "sync:s3cret"
    assert fake.headers == [{"Authorization": "Basic c3luYzpzM2NyZXQ="}] * 3


def test_sync_restores_fresh_grafana(grafana):
    # a redeployed Grafana: only a label drawn before the first sync
    fake = grafana([
        annotation(1, "Station fault", ["label:instrument", RU], time=1647244800000),
    ])
    stored = [stored_label("a1", ["label:noise", RU]),
              stored_label("p1", ["label:tremor", RU], time_end=None)]

    labels, report = grafana_annotations.sync_labels("http://grafana", stored)

    assert (report["restored"], report["added"]) == (2, 1)
    posted = {grafana_annotations._split_id(p["text"])[1]: p for p in fake.posted}
    assert sorted(posted) == ["a1", "p1"]
    # organization annotations: no dashboard, shown wherever the volcano tag is queried
    assert posted["a1"] == {"time": 1647158400000, "timeEnd": 1647165600000,
                            "text": "Wind noise\nGusts\nId: a1",
                            "tags": ["label:noise", RU]}
    assert posted["p1"]["timeEnd"] == posted["p1"]["time"]
    assert len(labels) == 3

    # the next sync finds every label in Grafana and changes nothing, and the
    # restored labels keep where they were originally made
    again, report = grafana_annotations.sync_labels("http://grafana", labels)
    assert (report["restored"], report["added"]) == (0, 0)
    assert len(fake.posted) == 2
    assert again == labels


def test_sync_cli(grafana, tmp_path, capsys):
    grafana([
        annotation(1, "Wind noise", ["label:noise", RU], time_end=1647165600000),
        annotation(2, "Forgot the volcano\nmore", ["label:noise"], time=1647244800000),
    ])
    path = tmp_path / "labels.json"

    grafana_annotations.main(["http://grafana", "--file", str(path)])

    labels = json.loads(path.read_text())
    assert [label["title"] for label in labels] == ["Wind noise", "Forgot the volcano"]
    assert not (tmp_path / "labels.json.tmp").exists()
    out = capsys.readouterr().out
    assert f"2 labels in {path} (0 restored to Grafana, 2 added)" in out
    assert "label:noise: 2" in out
    assert "1 labels have no volcano: tag" in out
    assert "2022-03-14 08:00 UTC on dashboard rsam: Forgot the volcano" in out

    # a second run reads the file back and changes nothing
    grafana_annotations.main(["http://grafana", "--file", str(path)])
    assert json.loads(path.read_text()) == labels
    assert "(0 restored to Grafana, 0 added)" in capsys.readouterr().out
