---
doc_id: ki-018-hearth-hub-dst-offset
title: "Known issue: Hearth Hub schedules one hour off after clock change"
type: known_issue
issue_id: KI-018
category: technical
regions: [US, CA, UK, EU]
updated: 2026-10-26
---
# Known issue: Hearth Hub schedules one hour off after clock change

## Symptoms
After the daylight saving time change, Hearth Hub schedules and automations run one hour early or late, for example lights turning on at 6:00 instead of 7:00. The time in the app is correct.

## Cause
Hubs on firmware 5.0.2 or earlier with a manually selected time zone do not apply the clock change.

## Resolution
In the QuillCloud app > Devices > Hearth Hub > Time zone, turn on "Set automatically", or select the time zone again. Schedules then run at the right time. Firmware 5.0.3 fixes this permanently.

## Status
Fixed in firmware 5.0.3.
