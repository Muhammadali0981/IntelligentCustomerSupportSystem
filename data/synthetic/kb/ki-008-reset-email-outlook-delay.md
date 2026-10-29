---
doc_id: ki-008-reset-email-outlook-delay
title: "Known issue: Password reset emails delayed for Outlook and Hotmail"
type: known_issue
issue_id: KI-008
category: account_access
regions: [US, CA, UK, EU]
updated: 2026-10-12
---
# Known issue: Password reset emails delayed for Outlook and Hotmail

## Symptoms
Customers with outlook.com, hotmail.com or live.com email addresses do not receive the password reset email, or it arrives 30-90 minutes late, by which time the link has expired.

## Cause
The email provider is temporarily slowing delivery of messages from noreply@quillfen.example.

## Resolution
Request one reset link only, then wait up to 90 minutes without requesting another (each new request cancels the earlier link). Check the Junk folder and add noreply@quillfen.example to safe senders. If the link expires, request a new one and use it as soon as it arrives. Alternatively, sign in with the QuillCloud app if still signed in and change the password under Account > Security.

## Status
Working with the email provider.
