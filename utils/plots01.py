# plot_hist.py
import json
import matplotlib.pyplot as plt

with open("times.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sys_t = data["system_clock_ms"]
st_t = data["steady_clock_ms"]
mach_t = data["mach_time_ms"]

# bins можно подбирать: например, 10 или sqrt(K)
bins = 100

fig, axs = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)

axs[0].hist(sys_t, bins=bins, edgecolor="black")
axs[0].set_title("system_clock (ms)")
axs[0].set_xlabel("ms")
axs[0].set_ylabel("count")

axs[1].hist(st_t, bins=bins, edgecolor="black")
axs[1].set_title("steady_clock (ms)")
axs[1].set_xlabel("ms")
axs[1].set_ylabel("count")

axs[2].hist(mach_t, bins=bins, edgecolor="black")
axs[2].set_title("mach_absolute_time (ms)")
axs[2].set_xlabel("ms")
axs[2].set_ylabel("count")

plt.show()
