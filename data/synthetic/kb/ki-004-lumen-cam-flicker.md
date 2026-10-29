---
doc_id: ki-004-lumen-cam-flicker
title: "Known issue: Lumen Cam image flickers after firmware 2.4 (UK/EU)"
type: known_issue
issue_id: KI-004
category: technical
regions: [UK, EU]
updated: 2026-09-12
---
# Known issue: Lumen Cam image flickers after firmware 2.4 (UK/EU)

## Symptoms
In the UK and EU, after the camera updated to firmware 2.4, the indoor video shows rolling dark bands or flicker under lamps, especially in the evening. Outdoor daylight video looks normal.

## Cause
Firmware 2.4 reset the Anti-flicker setting to 60 Hz on all cameras. UK and EU mains power is 50 Hz.

## Resolution
In the QuillCloud app > Lumen Cam > Video, set Anti-flicker to 50 Hz. The setting is kept after future updates.

## Status
Firmware 2.4.1 restores the correct regional default for new set-ups.
