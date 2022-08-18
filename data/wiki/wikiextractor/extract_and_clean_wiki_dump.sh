#!/bin/sh
set -e

WIKI_DUMP_FILE_IN=$1
WIKI_DUMP_FILE_OUT=${WIKI_DUMP_FILE_IN%%.*}.txt

# clone the WikiExtractor repository
#git clone https://github.com/attardi/wikiextractor.git

# extract and clean the chosen Wikipedia dump
echo "Extracting and cleaning $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT..."
python3 wikiextractor/WikiExtractor.py $WIKI_DUMP_FILE_IN --processes 32 --json -q -o - \
| sed "/^\s*\$/d" \
| grep -v "^<doc id=" \
| grep -v "</doc>\$" \
> $WIKI_DUMP_FILE_OUT
echo "Succesfully extracted and cleaned $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT"
