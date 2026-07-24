#!/bin/bash
# start_display.sh -- bring up the X/VNC stack, then run the given command.
#
# Deliberately boring. Uses ONLY bash builtins + sleep + test -- no bc, no xdpyinfo, no
# /dev/tcp probes. An earlier version depended on xdpyinfo (NOT installed in this image) and
# on bc for its timeout arithmetic, which produced a silent infinite loop.
#
# The sleep timings below are the ones that worked for every earlier exp2 run in this same
# image. The only addition is a BOUNDED wait on the X socket -- bounded so that worst case it
# gives up and tells you why, instead of hanging forever.
#
# USAGE: bash start_display.sh <command> [args...]

DISP=:99

echo "[display] starting Xvfb on $DISP"
Xvfb $DISP -screen 0 1920x1080x24 -ac >/tmp/xvfb.log 2>&1 &

# bounded wait for the X socket: 60 x 0.25s = 15s max. Pure bash, no external tools.
ready=0
for i in $(seq 1 60); do
    if [ -e /tmp/.X11-unix/X99 ]; then ready=1; break; fi
    sleep 0.25
done
if [ "$ready" = "1" ]; then
    echo "[display] Xvfb ready"
else
    echo "[display] WARNING: X socket /tmp/.X11-unix/X99 never appeared after 15s."
    echo "[display] Xvfb log:"; tail -20 /tmp/xvfb.log 2>/dev/null
    echo "[display] continuing anyway -- the command may fail, but you will see the real error."
fi

echo "[display] starting fluxbox"
DISPLAY=$DISP fluxbox >/tmp/fluxbox.log 2>&1 &
sleep 1

echo "[display] starting x11vnc"
x11vnc -display $DISP -nopw -forever -rfbport 5900 -cursor arrow >/tmp/x11vnc.log 2>&1 &
sleep 1

echo "[display] starting websockify (noVNC on :8080)"
python3 -m websockify --web /opt/noVNC 0.0.0.0:8080 localhost:5900 >/tmp/websockify.log 2>&1 &
sleep 2

echo "[display] launching: $*"
echo "----------------------------------------------------------------------"
DISPLAY=$DISP exec "$@"

