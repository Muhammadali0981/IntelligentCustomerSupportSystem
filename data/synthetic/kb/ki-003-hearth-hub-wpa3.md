---
doc_id: ki-003-hearth-hub-wpa3
title: "Known issue: Hearth Hub offline on WPA3-only networks"
type: known_issue
issue_id: KI-003
category: technical
regions: [US, CA, UK, EU]
updated: 2026-10-10
---
# Known issue: Hearth Hub offline on WPA3-only networks

## Symptoms
The Hearth Hub goes offline (pulsing orange light) after the router's security setting changed to WPA3-only, often after an automatic router update or a new router from the internet provider. Setting the hub up again fails at the Wi-Fi step.

## Cause
Hearth Hub firmware before 5.1 supports WPA2 and WPA2/WPA3 mixed mode, but not WPA3-only.

## Resolution
In the router settings, set Wi-Fi security to "WPA2/WPA3 mixed" (sometimes called "WPA3 transition mode"). The hub then reconnects and updates itself to firmware 5.1, after which WPA3-only also works.

## Status
Firmware 5.1 rolling out in October 2026.
