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
        # Draw on a copy so we don't overwrite the original frame permanently
        temp_frame = frame.copy()
        for pt in roiPts:
            cv2.circle(temp_frame, pt, 4, (0, 255, 0), 2)
        cv2.imshow("frame", temp_frame)


def main():
    global frame, roiPts, inputMode

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    args = vars(ap.parse_args())

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

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", selectROI)

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    roiHist = None

    try:
        while True:
            grabbed, frame = camera.read()

            if not grabbed or frame is None:
                print("Failed to grab frame or end of video reached")
                break

            if roiBox is not None and roiHist is not None:
                try:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
                    r, roiBox = cv2.CamShift(backProj, roiBox, termination)
                    pts = cv2.boxPoints(r).astype(int)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                except cv2.error as e:
                    print(f"Tracking error: {e}")
                    roiBox = None  # Reset tracking on error

            cv2.imshow("frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("i") and len(roiPts) < 4:
                inputMode = True
                orig = frame.copy()

                # Wait for 4 points to be selected
                while len(roiPts) < 4:
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(0) == ord("q"):
                        raise KeyboardInterrupt

                try:
                    roiPtsArr = np.array(roiPts)
                    if len(roiPtsArr) != 4:
                        raise ValueError("Need exactly 4 points for ROI")

                    s = roiPtsArr.sum(axis=1)
                    tl = roiPtsArr[np.argmin(s)]
                    br = roiPtsArr[np.argmax(s)]

                    if tl[0] >= br[0] or tl[1] >= br[1]:
                        raise ValueError("Invalid ROI coordinates")

                    roi = orig[tl[1]:br[1], tl[0]:br[0]]
                    if roi.size == 0:
                        raise ValueError("Empty ROI selected")

                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
                    roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
                    roiBox = (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])  # Fix width and height

                except Exception as e:
                    print(f"ROI selection error: {e}")
                    roiPts = []
                    roiBox = None
                    roiHist = None

                inputMode = False
                roiPts = []  # Clear points after selection

            elif key == ord("q"):
                break

    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
