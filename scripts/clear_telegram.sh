#!/bin/bash
set -e

telegramfolder=$(find ~/Library/Group\ Containers -type d -maxdepth 1 -name "*.keepcoder.Telegram")
telegramaccountfolder=$(find "${telegramfolder}" -type d -maxdepth 1 -name "account-*")
rm -r "${telegramaccountfolder}/postbox/media"
