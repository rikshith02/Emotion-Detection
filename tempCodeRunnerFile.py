
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Exit on ESC
            break
    except cv2.error:
        pass

# Release webcam and close OpenCV windows
webcam.release()
cv2.destroyAllWindows()

