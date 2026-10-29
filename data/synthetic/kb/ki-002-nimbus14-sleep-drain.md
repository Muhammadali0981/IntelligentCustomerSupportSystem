---
doc_id: ki-002-nimbus14-sleep-drain
title: "Known issue: Nimbus 14 battery drains in sleep (BIOS 1.08)"
type: known_issue
issue_id: KI-002
category: technical
regions: [US, CA, UK, EU]
updated: 2026-09-05
---
# Known issue: Nimbus 14 battery drains in sleep (BIOS 1.08)

## Symptoms
Nimbus 14 laptops on BIOS 1.08 lose 20-40% battery overnight with the lid closed, and may feel warm in a bag. Battery life while in use is normal.

## Cause
BIOS 1.08 prevents the laptop from entering its deepest sleep state.

## Resolution
Update the BIOS to version 1.10 with Quillfen Update Utility > BIOS, with the charger connected. Until then, use Hibernate instead of Sleep. Battery health is not affected; no warranty claim is needed.

## Status
Fixed in BIOS 1.10, released 5 September 2026.
