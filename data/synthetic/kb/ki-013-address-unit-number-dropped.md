---
doc_id: ki-013-address-unit-number-dropped
title: "Known issue: Address autocomplete drops apartment or unit number"
type: known_issue
issue_id: KI-013
category: shipping
regions: [US, CA, UK, EU]
updated: 2026-09-15
---
# Known issue: Address autocomplete drops apartment or unit number

## Symptoms
When customers pick an address from the autocomplete suggestions at checkout, the apartment, unit or flat number they typed is removed. The order confirmation shows the street address without it.

## Cause
A bug in the address suggestion tool.

## Resolution
Within 1 hour of ordering, the customer can add the unit number in Account > Orders > Edit order. After 1 hour, support adds the unit number for the carrier as a delivery note if the order has not shipped. If it has shipped, US customers can add it through the carrier redirect; elsewhere contact the carrier with the tracking number.

## Status
Fix planned; workaround is to type the full address without choosing a suggestion.
