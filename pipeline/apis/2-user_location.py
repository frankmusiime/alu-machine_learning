#!/usr/bin/env python3
"""Display a GitHub user's location."""

import sys
import time

import requests


if __name__ == "__main__":
    res = requests.get(sys.argv[1])

    if res.status_code == 403:
        reset = int(res.headers.get("X-Ratelimit-Reset", 0))
        current = int(time.time())
        minutes = (reset - current) // 60
        print("Reset in {} min".format(minutes))

    elif res.status_code == 404:
        print("Not found")

    elif res.status_code == 200:
        data = res.json()
        print(data.get("location"))
