#!/usr/bin/env python3
"""
Plot network throughput over time from one or more pcap files.

Usage:
    python3 plot_throughput.py run1.pcap
    python3 plot_throughput.py selfhealing.pcap original.pcap   # overlay for comparison

Requires: tshark (installed with wireshark), pandas, matplotlib
    pip install pandas matplotlib --break-system-packages
"""
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt

BIN_SECONDS = 1.0  # width of each time bucket, in seconds


def extract(pcap):
    """Return a Series of throughput (Mbps) indexed by time bucket."""
    # Pull the relative timestamp and byte length of every frame
    result = subprocess.run(
        ["tshark", "-r", pcap, "-T", "fields",
         "-e", "frame.time_relative", "-e", "frame.len"],
        capture_output=True, text=True, check=True
    )

    rows = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[0] and parts[1]:
            rows.append((float(parts[0]), int(parts[1])))

    if not rows:
        raise ValueError(f"No packets found in {pcap}")

    df = pd.DataFrame(rows, columns=["t", "bytes"])
    # Drop each packet into a time bucket, then sum the bytes in each bucket
    df["bucket"] = (df["t"] // BIN_SECONDS) * BIN_SECONDS
    bytes_per_bin = df.groupby("bucket")["bytes"].sum()
    # bytes per bin  ->  megabits per second
    mbps = (bytes_per_bin * 8) / (BIN_SECONDS * 1_000_000)
    return mbps


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_throughput.py <file1.pcap> [file2.pcap ...]")
        sys.exit(1)

    plt.figure(figsize=(10, 5))
    for pcap in sys.argv[1:]:
        mbps = extract(pcap)
        plt.plot(mbps.index, mbps.values, label=pcap, linewidth=1.5)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Throughput (Mbps)")
    plt.title("NetGent Throughput Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("throughput2.png", dpi=150)
    print("Saved throughput.png")


if __name__ == "__main__":
    main()

