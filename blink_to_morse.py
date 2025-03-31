import numpy as np
import argparse
import cv2
import sys

# Initialize global variables
frame = None
roiPts = []
inputMode = False


def selectROI(event, x, y, flags, param):
    global frame, roiPts, inputMode

    if frame is None:  # Safety check
        return

    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)


def main():
    global frame, roiPts, inputMode

    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    args = vars(ap.parse_args())

    # Initialize video capture with error handling
    try:
        if not args.get("video", False):
            camera = cv2.VideoCapture(0)  # Webcam
            if not camera.isOpened():
                raise ValueError("Could not open webcam")
        else:
            camera = cv2.VideoCapture(args["video"])  # Video file
            if not camera.isOpened():
                raise ValueError(f"Could not open video file: {args['video']}")
    except Exception as e:
        print(f"Error initializing video capture: {e}")
        sys.exit(1)

    # Set up mouse callback
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", selectROI)

    # Initialize tracking variables
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    roiHist = None

    try:
        while True:
            (grabbed, frame) = camera.read()

            if not grabbed or frame is None:
                print("Failed to grab frame or end of video reached")
                break

            # If ROI is selected, perform tracking
            if roiBox is not None and roiHist is not None:
                try:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
                    (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
                    pts = cv2.boxPoints(r).astype(int)  # Fixed: Replace np.int0 with astype(int)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                except cv2.error as e:
                    print(f"Tracking error: {e}")
                    roiBox = None  # Reset tracking on error

            cv2.imshow("frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # Enter ROI selection mode
            if key == ord("i") and len(roiPts) < 4:
                inputMode = True
                orig = frame.copy()

                while len(roiPts) < 4:
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(0) == ord("q"):  # Allow exit during selection
                        raise KeyboardInterrupt

                try:
                    # Calculate ROI bounding box
                    roiPts = np.array(roiPts)
                    if len(roiPts) != 4:
                        raise ValueError("Need exactly 4 points for ROI")

                    s = roiPts.sum(axis=1)
                    tl = roiPts[np.argmin(s)]
                    br = roiPts[np.argmax(s)]

                    # Validate ROI coordinates
                    if tl[0] >= br[0] or tl[1] >= br[1]:
                        raise ValueError("Invalid ROI coordinates")

                    # Compute ROI histogram
                    roi = orig[tl[1]:br[1], tl[0]:br[0]]
                    if roi.size == 0:
                        raise ValueError("Empty ROI selected")

                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
                    roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
                    roiBox = (tl[0], tl[1], br[0], br[1])

                except Exception as e:
                    print(f"ROI selection error: {e}")
                    roiPts = []
                    roiBox = None
                    roiHist = None

                inputMode = False

            # Quit program
            elif key == ord("q"):
                break

    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
