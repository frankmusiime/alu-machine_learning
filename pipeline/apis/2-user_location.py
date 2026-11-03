
#!/usr/bin/env python3
"""Display a GitHub user's location."""

import requests
import sys
import time
import math


if __name__ == "__main__":
    res = requests.get(sys.argv[1])

    if res.status_code == 403:
        reset_time = int(res.headers.get("X-Ratelimit-Reset", 0))
        current_time = time.time()

        minutes = max(0, math.ceil((reset_time - current_time) / 60))
        print("Reset in {} min".format(minutes))

    elif res.status_code == 404:
        print("Not found")

    elif res.status_code == 200:
        data = res.json()
        print(data.get("location"))
