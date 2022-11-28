import cv2
import numpy as np
import sys
import os

def main():
    if (len(sys.argv) != 3):
        print("Not enough params:")
        print('python3 mp4_to_frames.py <filename> <output_folder>')
    
    filename = sys.argv[1]
    outfolder = sys.argv[2]
    os.mkdir(outfolder)
    cap = cv2.VideoCapture(filename)
    i = 0
    while (cap.isOpened()):
        print(f'writing frame {i}')
        outfile = f"{outfolder}/frame_{i}"
        i += 1
        is_read, frame = cap.read()
        if not is_read:
            break
        cv2.imwrite(outfile, frame)
        print(f"Wrote image {outfile}")
    cap.release()
    print("Done")

if __name__ == "__main__":
    main()
