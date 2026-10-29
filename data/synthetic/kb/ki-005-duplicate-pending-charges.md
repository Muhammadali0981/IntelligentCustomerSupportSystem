---
doc_id: ki-005-duplicate-pending-charges
title: "Known issue: Duplicate pending charges after retrying payment"
type: known_issue
issue_id: KI-005
category: billing
regions: [US, CA, UK, EU]
updated: 2026-08-28
---
# Known issue: Duplicate pending charges after retrying payment

## Symptoms
Customers see two or three identical pending charges for one order on their card or banking app, usually after the first checkout attempt showed an error and they tried again. Only one order appears in their account.

## Cause
Each checkout attempt places a temporary authorisation. Failed attempts are not charged, but the bank shows the hold until it expires.

## Resolution
Only one charge is taken. The extra pending authorisations disappear automatically within 3-7 business days, depending on the bank. No action is needed; we cannot release holds earlier. If an extra amount is still shown as a completed (not pending) charge after 7 business days, escalate to the billing team with the order number.

## Status
Ongoing; caused by how banks show authorisations.
