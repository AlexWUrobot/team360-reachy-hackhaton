def main() -> None:
    # On Linux, combining ReachyMini's in-process GStreamer/GLib camera pipeline
    # (mini.media.get_frame) with OpenCV's Qt windows (imshow) often triggers
    # thread/timer warnings like:
    # - g_main_context_push_thread_default: 'acquired_context' failed
    # - QObject::startTimer / QObject::killTimer
    # and the UI becomes unstable.
    #
    # The most robust approach is to read the Reachy Mini camera via V4L2
    # (/dev/video4 on your machine) and use the daemon only for robot control.
    from demo_reachy_v4l2_imshow_head_ears import main as v4l2_main

    v4l2_main()


if __name__ == "__main__":
    main()
