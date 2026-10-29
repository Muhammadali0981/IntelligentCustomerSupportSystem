---
doc_id: ki-009-app-signout-loop
title: "Known issue: QuillCloud app 4.0 keeps signing users out"
type: known_issue
issue_id: KI-009
category: account_access
regions: [US, CA, UK, EU]
updated: 2026-10-08
---
# Known issue: QuillCloud app 4.0 keeps signing users out

## Symptoms
After updating the QuillCloud app to version 4.0, users are signed out every few minutes or each time they open the app, and see "Session expired". The website works normally. Devices keep working.

## Cause
A bug in app version 4.0 when refreshing the sign-in session.

## Resolution
Update the app to version 4.0.2 or later. On Android, clear the app cache after updating; on iPhone, delete and reinstall the app. Then sign in again. The password does not need to be reset.

## Status
Fixed in app version 4.0.2.
