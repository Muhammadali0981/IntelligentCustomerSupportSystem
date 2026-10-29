---
doc_id: ki-006-promo-code-plus-members
title: "Known issue: Promo code shows 'invalid' for Plus members"
type: known_issue
issue_id: KI-006
category: billing
regions: [US, CA, UK, EU]
updated: 2026-10-03
---
# Known issue: Promo code shows "invalid" for Plus members

## Symptoms
Quillfen Plus members entering a valid, unexpired promo code see the error "This code is not valid" at checkout, mostly on orders of Quillfen-brand products.

## Cause
Promo codes do not combine with the 5% Plus member price. Checkout should show "Better discount applied" when the member price is larger, but since 1 October 2026 it shows the wrong "not valid" message instead.

## Resolution
Explain that the larger discount is already applied: the member price is in the order total. If the promo code would have given a bigger discount than the member price, the customer can place the order and support will refund the difference on request.

## Status
Message fix planned for the November website release.
