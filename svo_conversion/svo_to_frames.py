import sys
import pyzed.sl as sl
import cv2
import numpy as np
from pathlib import Path

def main():

    if len(sys.argv) < 3:
        print("Please specify path to .svo file and output .mp4 file.")
        print("Usage: python3 sov_to_frames.py <inputfile.svo> <outfolder> [name_head] [index] [png/jpg]")
        exit()

    filepath = sys.argv[1]
    outfilepath = sys.argv[2]
    i = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    filename = sys.argv[4] if len(sys.argv) > 4 else "frame"
    filetype = sys.argv[5] if len(sys.argv) > 5 else "jpg"

    Path(outfilepath).mkdir(parents=True, exist_ok=True)

    print("Reading SVO file: {0}".format(filepath))
    print("Writing to files: {0}/{1}_X".format(outfilepath, filename))

    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print('exception!')
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    # Read in the file and write it to another
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 30
    err = cam.grab(runtime)
    if err != sl.ERROR_CODE.SUCCESS:
        print('Error reading first frame')
        return
    
    cam.retrieve_image(mat)
    frame = mat.get_data()
    print(frame)
    print(f"Original frame shape: {frame.shape}")
    out_shape = (frame.shape[0], frame.shape[1], 3)
    frame = np.resize(frame, out_shape)
    frame_width = frame.shape[0]
    frame_height = frame.shape[1]
    print(f"target frame shape: {frame.shape}")
    print(f"Using frame size of ({frame_width}, {frame_height} with FPS {fps}")
    #output = cv2.VideoWriter(outfilepath, fourcc, fps, (frame_height, frame_width), isColor=True)
    #output.write(frame)
    cv2.imwrite(f"{outfilepath}/{filename}.{filetype}_{i}", frame)
    #i = 0
    i += 1
    while True:
        err = cam.grab(runtime)
        outfile = f"{outfilepath}/{filename}.{filetype}_{i}"
        i += 1
        if err != sl.ERROR_CODE.SUCCESS or i == 5:
            print('Error reading image, ending program')
            #output.release()
            cam.close()
            return
        cam.retrieve_image(mat)
        frame = mat.get_data()
        frame = np.resize(frame, out_shape)
        print(f"Reading frame {i} of size {frame.shape}")
        #output.write(frame)
        cv2.imwrite(outfile, frame)

    return # Do not run code after this

    key = ''
    print("  Save the current image:     s")
    print("  Quit the video reading:     q\n")
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat)
            cv2.imshow("ZED", mat.get_data())
            key = cv2.waitKey(1)
            saving_image(key, mat)
        else:
            key = cv2.waitKey(1)
    cv2.destroyAllWindows()

    print_camera_information(cam)
    cam.close()
    print("\nFINISH")


def print_camera_information(cam):
    while True:
        res = input("Do you want to display camera information? [y/n]: ")
        if res == "y":
            print()
            print("Distorsion factor of the right cam before calibration: {0}.".format(
                cam.get_camera_information().calibration_parameters_raw.right_cam.disto))
            print("Distorsion factor of the right cam after calibration: {0}.\n".format(
                cam.get_camera_information().calibration_parameters.right_cam.disto))

            print("Confidence threshold: {0}".format(cam.get_runtime_parameters().confidence_threshold))
            print("Depth min and max range values: {0}, {1}".format(cam.get_init_parameters().depth_minimum_distance,
                                                                    cam.get_init_parameters().depth_maximum_distance)
)
            print("Resolution: {0}, {1}.".format(round(cam.get_camera_information().camera_resolution.width, 2), cam.get_camera_information().camera_resolution.height))
            print("Camera FPS: {0}".format(cam.get_camera_information().camera_fps))
            print("Frame count: {0}.\n".format(cam.get_svo_number_of_frames()))
            break
        elif res == "n":
            print("Camera information not displayed.\n")
            break
        else:
            print("Error, please enter [y/n].\n")


def saving_image(key, mat):
    if key == 115:
        img = sl.ERROR_CODE.FAILURE
        while img != sl.ERROR_CODE.SUCCESS:
            filepath = input("Enter filepath name: ")
            img = mat.write(filepath)
            print("Saving image : {0}".format(repr(img)))
            if img == sl.ERROR_CODE.SUCCESS:
                break
            else:
                print("Help: you must enter the filepath + filename + PNG extension.")



if __name__ == "__main__":
    main()
