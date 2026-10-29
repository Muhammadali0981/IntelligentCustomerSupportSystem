---
doc_id: ki-015-nimbus14-monitor-driver
title: "Known issue: Nimbus 14 external monitor black after graphics driver 31.0.15"
type: known_issue
issue_id: KI-015
category: technical
regions: [US, CA, UK, EU]
updated: 2026-10-14
---
# Known issue: Nimbus 14 external monitor black after graphics driver 31.0.15

## Symptoms
After the graphics driver updated to 31.0.15, an external monitor connected to the Nimbus 14's left USB-C port stays black or shows "No signal". The laptop detects the monitor in Display settings. The built-in screen works.

## Cause
Driver 31.0.15 has a bug with DisplayPort Alt Mode on the Nimbus 14.

## Resolution
Install graphics driver 31.0.17 from Quillfen Update Utility > Drivers and restart. Keep using the left USB-C port; the right port never supports video.

## Status
Fixed in driver 31.0.17.
