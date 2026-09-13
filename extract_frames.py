"""
Extract ALL 240 frames from the ocean scroll video as WebP.
More frames = smaller visual gap between each cut = smoother animation.
Output: public/frames/f000.webp ... f239.webp
"""
import cv2, os, sys

SRC  = r"c:\Users\sarke\Desktop\SIh prototype\public\ocean_bg.mp4"
OUT  = r"c:\Users\sarke\Desktop\SIh prototype\public\frames"
QUAL = 85   # WebP quality

os.makedirs(OUT, exist_ok=True)

cap   = cv2.VideoCapture(SRC)
if not cap.isOpened():
    sys.exit("ERROR: cannot open " + SRC)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Video:", total, "frames |", round(fps,1), "fps |", w, "x", h)

N = total   # export EVERY frame

ok_count = 0
for i in range(N):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        print("  WARN: skip frame", i)
        continue
    out_path = os.path.join(OUT, "f" + str(i).zfill(3) + ".webp")
    cv2.imwrite(out_path, frame, [cv2.IMWRITE_WEBP_QUALITY, QUAL])
    ok_count += 1
    if i % 30 == 0:
        print("  [" + str(i+1) + "/" + str(N) + "] written")

cap.release()
print("Done:", ok_count, "/", N, "frames ->", OUT)
