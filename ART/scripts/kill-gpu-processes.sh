#!/bin/bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9