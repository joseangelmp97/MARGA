"""
Main application loop and CLI entrypoint.

"""

import time
from .config import AppConfig, default_config
from .builders import build_source, build_detector, build_hazard_logic, build_notifier


# Main loop

def run(cfg: AppConfig) -> None:
    """
    Main application loop.

    This function orchestrates the entire application. It initializes the modules based on the configuration, then enters a loop where it continuously gets frames from the source, runs detection, updates hazard state, and sends notifications. It also handles graceful shutdown when the source is exhausted or an error occurs.
    
    :param cfg: Description
    :type cfg: AppConfig
    """
    source = build_source(cfg)
    detector = build_detector(cfg)
    hazard_logic = build_hazard_logic(cfg)
    notifier = build_notifier(cfg)

    print("MVP pipeline started.")
    print(f"- Source:   {cfg.capture.source_type}")
    print(f"- Detector: {cfg.detector.detector_type}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            #1) Capture frame
            packet = source.get_frame_packet()
            if packet is None:
                # Stream can temporarily fail; do not crash the whole app.
                print("No frame available.")
                time.sleep(0.1)
                continue
            #2) Perception
            detections = detector.detect(packet)
            #3) Decision
            hazard_state = hazard_logic.update(packet, detections)
            #4) Feedback
            notifier.notify(packet, hazard_state)

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        #Ensure we release resources even on crashes / ctrl+c
        try:
            notifier.close()
        except Exception:
            pass

        try:
            source.close()
        except Exception:
            pass
        print("Shutdown complete.")

def main() -> None:
    """
    CLI entrypoint. For now we just use default_config().
    Later we can add argparse for overrides.
    """
    cfg = default_config()
    run(cfg)


if __name__ == "__main__":
    main()