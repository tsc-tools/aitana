"""
Sync the labels people draw on Grafana dashboards with a label file.

Every annotation tagged `label:<class>` is a label, whichever dashboard it was drawn on,
and all of them are kept in one JSON file. What a label refers to is in its tags only:
`volcano:<name>`, and `series:<site>.<loc>.<cha>.<feature>` for a label that concerns
particular series rather than the whole volcano. Each label carries a stable id in an
"Id: <id>" line at the end of its text. The sync is a union: edits made in Grafana win,
and stored labels missing from Grafana (deleted there, or a freshly deployed Grafana) are
posted back as organization annotations, which any dashboard shows through an annotation
query on the volcano tag. To withdraw a label, retag it, e.g. `label:retracted`.
"""

import json
import os
import re
import uuid
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import requests

ID_LINE = re.compile(r"(?:^|\n)Id: (\S+)\s*$")


def is_label(tags: list[str]) -> bool:
    """Whether tags mark an annotation as a `label:` label."""
    return any(tag.startswith("label:") for tag in tags)


def _split_id(text: str):
    """Split annotation text into the text before its "Id: <id>" line and the id."""
    match = ID_LINE.search(text)
    if match is None:
        return text, None
    return text[: match.start()], match.group(1)


def _check(rval, action: str):
    if rval.status_code != 200:
        raise RuntimeError(f"Failed to {action}: {rval.status_code} {rval.text}")


def sync_labels(baseurl: str, stored: list[dict[str, Any]]):
    """
    Bring the label annotations in Grafana and the stored labels in step.

    The result is the union of both. A label in both takes its text, tags and
    times from Grafana; labels drawn in Grafana, on any dashboard, are added;
    stored labels missing from Grafana are posted as organization annotations,
    so nothing is lost when an annotation is deleted or Grafana is redeployed.
    Annotations without an "Id: <id>" line get a new id appended to their text.

    Parameters
    ----------
    baseurl : str
        The base URL of the Grafana API, e.g.
        "http://<grafana-user>:<user-pwd>@<grafana-host>:<port>".
    stored : list of dict
        Labels from the previous sync, as returned by this function.

    Returns
    -------
    labels : list of dict
        All labels, ordered by time. Each has 'time' and 'timeEnd' (ms since
        epoch; 'timeEnd' is None for a point), 'title' (first line of the
        text), 'description' (the text without its "Id:" line), 'tags', 'id'
        (the stable label id), and where they were made: 'user' (login),
        'dashboardUID' and 'panelId' (None for an annotation made through the
        API).
    report : dict
        Number of labels 'restored' to Grafana and 'added' from Grafana, and
        'untagged': the labels without a `volcano:` tag, which no volcano
        dashboard shows.
    """
    url = baseurl + "/api/annotations"
    rval = requests.get(url, params={"type": "annotation", "limit": 100000})
    _check(rval, "retrieve annotations")
    known = {label["id"]: label for label in stored}

    in_grafana = {}
    for atn in rval.json():
        tags = atn.get("tags", [])
        if not is_label(tags):
            continue
        description, label_id = _split_id(atn.get("text", ""))
        if label_id is None:
            label_id = uuid.uuid4().hex
            _check(
                requests.patch(
                    f"{url}/{atn['id']}",
                    json={"text": f"{description}\nId: {label_id}"},
                ),
                "add an id to an annotation",
            )
        # where a label was made survives its restore as an organization annotation
        origin = known.get(label_id, {})
        time_end = atn.get("timeEnd")
        in_grafana[label_id] = {
            "time": atn["time"],
            "timeEnd": time_end if time_end and time_end != atn["time"] else None,
            "title": description.split("\n")[0],
            "description": description,
            "tags": tags,
            "id": label_id,
            "user": origin.get("user") or atn.get("login"),
            "dashboardUID": origin.get("dashboardUID")
            or atn.get("dashboardUID")
            or None,
            "panelId": origin.get("panelId") or atn.get("panelId") or None,
        }

    missing = known.keys() - in_grafana.keys()
    for label_id in missing:
        label = known[label_id]
        _check(
            requests.post(
                url,
                json={
                    "time": label["time"],
                    "timeEnd": label["timeEnd"] or label["time"],
                    "text": f"{label['description']}\nId: {label_id}",
                    "tags": label["tags"],
                },
            ),
            "restore an annotation",
        )
        in_grafana[label_id] = label

    labels = sorted(in_grafana.values(), key=lambda label: (label["time"], label["id"]))
    report = {
        "restored": len(missing),
        "added": len(in_grafana.keys() - known.keys()),
        "untagged": [
            label
            for label in labels
            if not any(tag.startswith("volcano:") for tag in label["tags"])
        ],
    }
    return labels, report


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", type=str, help="Grafana API URL for annotations.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Label file, e.g. /data/labels.json. Created by the first sync.",
    )
    args = parser.parse_args(argv)
    try:
        with open(args.file) as fh:
            stored = json.load(fh)
    except FileNotFoundError:
        stored = []
    labels, report = sync_labels(args.url, stored)
    # write and rename, so an interrupted sync can't leave a truncated file
    tmp = args.file + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(labels, fh, indent=1)
    os.replace(tmp, args.file)

    counts = Counter(tag for label in labels for tag in label["tags"])
    print(
        f"{len(labels)} labels in {args.file} ({report['restored']} restored to "
        f"Grafana, {report['added']} added). Tags:"
    )
    for tag, n in sorted(counts.items()):
        print(f"  {tag}: {n}")
    if report["untagged"]:
        print(
            f"{len(report['untagged'])} labels have no volcano: tag, so no volcano "
            "dashboard shows them:"
        )
        for label in report["untagged"]:
            start = datetime.fromtimestamp(label["time"] / 1000, tz=UTC)
            print(
                f"  {start:%Y-%m-%d %H:%M} UTC on dashboard "
                f"{label['dashboardUID'] or '-'}: {label['title']}"
            )


if __name__ == "__main__":
    main()
