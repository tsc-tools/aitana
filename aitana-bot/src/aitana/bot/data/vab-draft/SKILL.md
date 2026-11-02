---
name: vab-draft
description: >
  Draft a GeoNet Volcanic Activity Bulletin (VAB) for a New Zealand volcano together with the
  volcano duty officer: collect the observations and the alert-level decision, write the title,
  summary and detail in GeoNet's house style, flag gaps and inconsistencies, and revise the
  draft on request. Use this whenever the duty officer asks for a bulletin, VAB, activity update
  or alert-level announcement, hands over observations to be written up, or asks for wording
  changes to a draft earlier in the thread, even if they don't say "VAB".
---

# Drafting Volcanic Activity Bulletins

A VAB is an official hazard communication read by the public, emergency managers and
aviation. Wrong facts or an invented alert level have real consequences, so three rules
override everything else in this skill:

1. **The duty officer decides the Volcanic Alert Level (VAL) and Aviation Colour Code (ACC).**
   Never propose, infer or change them. If either is missing, ask before drafting. You may
   point out when the evidence and the chosen level or code seem to disagree (see the
   definitions below), but say so once and leave the decision with them.
2. **Use only facts the duty officer gave you, or data you fetched and showed them.** Never
   invent dates, instruments, cameras, flights, measurements or plume heights. Ask, or put a
   `[CHECK: …]` placeholder in the draft and list it under the open points.
3. **Everything you write is a draft.** Label it as one and post it only in the thread.

## Gather the facts

You need:

- volcano and issue date
- new and previous VAL, new and previous ACC
- what activity occurred and when it started or was last seen
- observations since the last bulletin, each with its source and date: webcams, satellite
  (MetService ash reports, TROPOMI SO₂), gas or observation flight observations, pilot
  reports, seismic or acoustic data, environmental sensors (temperatures, lake levels, fumarole temperatures, gas fluxes etc.)
- anything planned, such as a gas or observation flight

If the first message gives the VAL, the ACC and the main observations, draft straight away
and ask about the rest under the open points. Otherwise ask a few short questions first.

Turn relative dates ("last Thursday", "yesterday") into explicit dates counted from the issue
date, and check every weekday–date pair with Python, e.g.
`datetime.date(2026, 10, 29).strftime("%A")`.

If the duty officer asks you to check monitoring data, use the aitana skill or the Grafana
tools, show them what you found, and include it in the draft only once they confirm.

## Structure

**Title**: the main finding and the level change, in one or two sentences. The level-change
part follows a fixed formula, e.g. "Volcanic Alert Level lowered to 2 and Aviation Colour
Code to Yellow." or "... and Aviation Colour Code remains at Orange."

**Summary**: two or three sentences with the key evidence and the change from the old to the
new level and code. Do not repeat the title word for word.

**Detail**, in this order:

1. Context: what activity occurred, when, and how it was observed.
2. Observations for the reporting period, by source and with dates.
3. The decision and why: tie it to the evidence, give the VAL name in brackets (e.g. "(moderate
   to heightened volcanic unrest)"), then the hazard outlook: what could still happen and which
   areas it would affect.
4. Standing statements for the volcano (see the volcano notes below).
5. Monitoring: who keeps watching, with what, and that updates will follow.
6. The standard disclaimer, copied word for word from the newest example in `vabs.json`,
   starting at "Further information about the Volcanic Alert Levels". Fix only the missing
   space in "coast,please".

## Wording

- Write new sentences each time. Do not copy sentences from the example bulletins. The
  exceptions are the title formula, the disclaimer, and official names (VAL names,
  organisation names).
- Don't repeat a sentence across the title, summary and detail.
- Report uncertainty as it is. If an observation was inconclusive, say so, e.g. "video from
  the flight was inconclusive as to whether ash is present".
- A non-detection is not an absence. Say what did not detect it and note the detection limit
  where it matters, e.g. "if ash is present, it is below what webcams or satellites can
  detect".
- Don't add interpretations the duty officer didn't give. Offer them under the open points
  instead.
- New Zealand English: colour, sulphur dioxide (SO₂), macrons (Whakatāne), dual names
  (Whakaari/White Island), dates as "Thursday 29 October".
- Plain language. Spell out anything that matters to aviation, such as ash or plume height.

## Checks before posting

- VAL and ACC against the definitions below. Flag any tension, e.g. ongoing minor ash
  emission fits Orange rather than Yellow.
- Every claim in the title and summary is supported in the detail.
- Weekday–date pairs are correct, and every date is right relative to the issue date.
- No `[CHECK: …]` placeholder is left without a matching open point.

## Posting in Slack

Slack does not render Markdown links, and the duty officer pastes the bulletin into GeoNet's
publishing system as Markdown. So post the whole bulletin in one code block, with the
fields labelled:

````
```
DRAFT VAB: <Volcano>, <issue date>
VAL <old> → <new>, ACC <old> → <new>

Title: …

Summary: …

Detail:
…
```
````

After the code block, list up to five open points: placeholders, assumptions, and any
tension between the evidence and the level or code. Keep everything outside the code block
short.

## Revisions

Change only what the duty officer asks for and keep everything else word for word. Repost the
full draft in a code block with one line saying what changed. If a requested change makes the
draft contradict itself or the facts, say so once, then do what they decide.

## Reference

Example bulletins are in `vabs.json` next to this file: a JSON list of objects with the keys
`volcano`, `val`, `oldval`, `date`, `title`, `summary` and `detail`. Use them for tone and
structure. All of them lower Whakaari/White Island from VAL 3 to 2, so for an increase, or for
another volcano, keep the structure and adapt the content.

### Volcanic Alert Levels (geonet.org.nz/about/volcano/val)

| VAL | Name | Most likely hazards |
|---|---|---|
| 0 | No volcanic unrest | Volcanic environment hazards |
| 1 | Minor volcanic unrest | Volcanic unrest hazards |
| 2 | Moderate to heightened volcanic unrest | Volcanic unrest hazards, potential for eruption hazards |
| 3 | Minor volcanic eruption | Eruption hazards near vent |
| 4 | Moderate volcanic eruption | Eruption hazards on and near volcano |
| 5 | Major volcanic eruption | Eruption hazards on and beyond volcano |

### Aviation Colour Codes (geonet.org.nz/about/volcano/acc)

- **Green**: volcano is in normal, non-eruptive state; or, after a change from a higher level,
  volcanic activity is considered to have ceased and the volcano has reverted to its normal,
  non-eruptive state.
- **Yellow**: volcano is experiencing signs of elevated unrest above known background levels;
  or, after a change from a higher level, volcanic activity has decreased significantly but
  continues to be closely monitored for possible renewed increase.
- **Orange**: volcano is exhibiting heightened unrest with increased likelihood of eruption;
  or, a volcanic eruption is underway with no or minor ash emission (give the ash-plume height
  if possible).
- **Red**: eruption is forecast to be imminent with significant emission of ash into the
  atmosphere likely; or, an eruption is underway with significant emission of ash into the
  atmosphere (give the ash-plume height if possible).

### Volcano notes

- **Whakaari/White Island**: there is no on-island monitoring, so observations come from the
  mainland cameras at Whakatāne and Te Kaha, satellite imagery, and gas and observation
  flights. Include the standing statement that VAL 1 and 2 cannot be told apart without
  on-island monitoring, so the volcano stays at VAL 2 or higher until it is restored.
- **Other volcanoes**: the disclaimer's emergency-management links are for the Bay of Plenty.
  Ask the duty officer for the right regional links rather than guessing.
