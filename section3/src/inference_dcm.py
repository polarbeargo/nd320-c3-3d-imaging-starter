"""
Here we do inference on a DICOM volume, constructing the volume first, and then sending it to the
clinical archive

This code will do the following:
    1. Identify the series to run HippoCrop.AI algorithm on from a folder containing multiple studies
    2. Construct a NumPy volume from a set of DICOM files
    3. Run inference on the constructed volume
    4. Create report from the inference
    5. Call a shell script to push report to the storage archive
"""

import os
import sys
import datetime
import time
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import pydicom

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from inference.UNetInferenceAgent import UNetInferenceAgent

def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects a Numpy array.
    Assumes that only one series is in the array

    Arguments:
        dcmlist {list of PyDicom objects} -- path to directory

    Returns:
        tuple of (3D volume, header of the 1st image)
    """

    # In the real world you would do a lot of validation here
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]

    # Make sure that you have correctly constructed the volume from your axial slices!
    hdr = dcmlist[0]

    # We return header so that we can inspect metadata properly.
    # Since for our purposes we are interested in "Series" header, we grab header of the
    # first file (assuming that any instance-specific values will be ighored - common approach)
    # We also zero-out Pixel Data since the users of this function are only interested in metadata
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)

def get_predicted_volumes(pred):
    """Gets volumes of two hippocampal structures from the predicted array

    Arguments:
        pred {Numpy array} -- array with labels. Assuming 0 is bg, 1 is anterior, 2 is posterior

    Returns:
        A dictionary with respective volumes
    """

    # TASK: Compute the volume of your hippocampal prediction
    # Count voxels for each class
    # Class 0: background
    # Class 1: anterior hippocampus
    # Class 2: posterior hippocampus
    
    volume_ant = np.sum(pred == 1)
    volume_post = np.sum(pred == 2)
    total_volume = volume_ant + volume_post

    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}

def process_single_slice_for_report(args):
    """Process a single slice for the report (normalization, coloring, compositing)
    
    Arguments:
        args: tuple of (orig_vol, pred_vol, slice_num, slice_index)
        
    Returns:
        tuple of (composite_image, slice_num, slice_index)
    """
    orig_vol, pred_vol, slice_num, slice_index = args
    
    try:
        orig_slice = orig_vol[:, :, slice_num]
        pred_slice = pred_vol[:, :, slice_num]
        
        orig_normalized = np.flip((orig_slice / np.max(orig_slice)) * 255).T.astype(np.uint8)
        pil_orig = Image.fromarray(orig_normalized, mode="L").convert("RGBA").resize((200, 200))
        
        # Create overlay by coloring the prediction mask
        # Anterior (class 1) = red, Posterior (class 2) = blue
        pred_colored = np.zeros((*pred_slice.shape, 4), dtype=np.uint8)
        pred_colored[pred_slice == 1] = [255, 0, 0, 100]
        pred_colored[pred_slice == 2] = [0, 0, 255, 100]
        pred_colored = np.flip(pred_colored, axis=0).transpose(1, 0, 2)
        
        pil_pred = Image.fromarray(pred_colored, mode="RGBA").resize((200, 200))
        composite = Image.alpha_composite(pil_orig, pil_pred)
        
        return (composite, slice_num, slice_index)
    except Exception as e:
        print(f"Error processing slice {slice_num}: {e}")
        return None

def create_report(inference, header, orig_vol, pred_vol):
    """Generates an image with inference report

    Arguments:
        inference {Dictionary} -- dict containing anterior, posterior and full volume values
        header {PyDicom Dataset} -- DICOM header
        orig_vol {Numpy array} -- original volume
        pred_vol {Numpy array} -- predicted label

    Returns:
        PIL image
    """

    # The code below uses PIL image library to compose an RGB image that will go into the report
    # A standard way of storing measurement data in DICOM archives is creating such report and
    # sending them on as Secondary Capture IODs (http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html)
    # Essentially, our report is just a standard RGB image, with some metadata, packed into 
    # DICOM format. 

    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)

    header_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=40)
    main_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=20)

    slice_nums = [orig_vol.shape[2]//3, orig_vol.shape[2]//2, orig_vol.shape[2]*3//4] # is there a better choice?

    # TASK: Create the report here and show information that you think would be relevant to
    # clinicians. A sample code is provided below, but feel free to use your creative 
    # genius to make if shine. After all, the is the only part of all our machine learning 
    # efforts that will be visible to the world. The usefulness of your computations will largely
    # depend on how you present them.
    draw.text((10, 0), "HippoVolume.AI", (255, 255, 255), font=header_font)
    draw.text((10, 50), "Automated Hippocampus Volume Report", (200, 200, 200), font=main_font)
    
    draw.multiline_text((10, 90),
                        f"Patient ID: {header.PatientID}\n"
                        f"Patient Name: {header.PatientName}\n"
                        f"Study Date: {header.StudyDate}\n"
                        f"Modality: {header.Modality}\n"
                        f"Series Description: {header.SeriesDescription}",
                        (255, 255, 255), font=main_font)
    
    draw.multiline_text((10, 220),
                        f"VOLUME MEASUREMENTS:\n"
                        f"Anterior Hippocampus: {inference['anterior']} voxels\n"
                        f"Posterior Hippocampus: {inference['posterior']} voxels\n"
                        f"Total Hippocampus: {inference['total']} voxels",
                        (100, 255, 100), font=main_font)
    
    slice_nums = [orig_vol.shape[2]//3, orig_vol.shape[2]//2, orig_vol.shape[2]*3//4]
    slice_args = [(orig_vol, pred_vol, slice_num, i) for i, slice_num in enumerate(slice_nums)]
    
    composite_images = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_single_slice_for_report, args) for args in slice_args]
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                composite, slice_num, slice_index = result
                composite_images[slice_index] = (composite, slice_num)
    
    for i in sorted(composite_images.keys()):
        composite, slice_num = composite_images[i]
        x_pos = 10 + (i * 210)
        y_pos = 360
        pimg.paste(composite, box=(x_pos, y_pos))
        draw.text((x_pos, y_pos + 210), f"Slice {slice_num}", (255, 255, 255), font=main_font)
    
    draw.text((10, 600), "LEGEND:", (255, 255, 255), font=main_font)
    draw.rectangle([10, 630, 30, 650], fill=(255, 0, 0))
    draw.text((40, 630), "Anterior Hippocampus", (255, 255, 255), font=main_font)
    draw.rectangle([10, 660, 30, 680], fill=(0, 0, 255))
    draw.text((40, 660), "Posterior Hippocampus", (255, 255, 255), font=main_font)
    
    draw.multiline_text((10, 720),
                        "DISCLAIMER: This report is generated by an AI algorithm\n"
                        "for research purposes. Clinical decisions should be made\n"
                        "by qualified healthcare professionals.",
                        (200, 200, 200), font=main_font)

    return pimg

def save_report_as_dcm(header, report, path):
    """Writes the supplied image as a DICOM Secondary Capture file

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        report {PIL image} -- image representing the report
        path {Where to save the report}

    Returns:
        N/A
    """

    # Code below creates a DICOM Secondary Capture instance that will be correctly
    # interpreted by most imaging viewers including our OHIF
    # The code here is complete as it is unlikely that as a data scientist you will 
    # have to dive that deep into generating DICOMs. However, if you still want to understand
    # the subject, there are some suggestions below

    # Set up DICOM metadata fields. Most of them will be the same as original file header
    out = pydicom.Dataset(header)

    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # STAND OUT SUGGESTION: 
    # If you want to understand better the generation of valid DICOM, remove everything below
    # and try writing your own DICOM generation code from scratch.
    # Refer to this part of the standard to see what are the requirements for the valid
    # Secondary Capture IOD: http://dicom.nema.org/medical/dicom/2019e/output/html/part03.html#sect_A.8
    # The Modules table (A.8-1) contains a list of modules with a notice which ones are mandatory (M)
    # and which ones are conditional (C) and which ones are user-optional (U)
    # Note that we are building an RGB image which would have three 8-bit samples per pixel
    # Also note that writing code that generates valid DICOM has a very calming effect
    # on mind and body :)

    out.is_little_endian = True
    out.is_implicit_VR = False

    # We need to change class to Secondary Capture
    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    # Our report is a separate image series of one image
    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "OT" # Other
    out.SeriesDescription = "HippoVolume.AI"

    out.Rows = report.height
    out.Columns = report.width

    out.ImageType = r"DERIVED\PRIMARY\AXIAL" # We are deriving this image from patient data
    out.SamplesPerPixel = 3 # we are building an RGB image.
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0 # means that bytes encode pixels as R1G1B1R2G2B2... as opposed to R1R2R3...G1G2G3...
    out.BitsAllocated = 8 # we are using 8 bits/pixel
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    # Set time and date
    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm

    out.ImagesInAcquisition = 1

    # We empty these since most viewers will then default to auto W/L
    out.WindowCenter = ""
    out.WindowWidth = ""

    # Data imprinted directly into image pixels is called "burned in annotation"
    out.BurnedInAnnotation = "YES"

    out.PixelData = report.tobytes()

    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)

def load_single_dicom_file(filepath):
    """Helper function to load a single DICOM file
    
    Arguments:
        filepath {string} -- full path to DICOM file
        
    Returns:
        PyDicom object or None if error
    """
    try:
        return pydicom.dcmread(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_series_for_inference(path):
    """Reads multiple series from one folder and picks the one
    to run inference on. Uses parallel processing for faster DICOM loading.

    Arguments:
        path {string} -- location of the DICOM files

    Returns:
        Numpy array representing the series
    """

    # Here we are assuming that path is a directory that contains a full study as a collection
    # of files
    # We are reading all files into a list of PyDicom objects so that we can filter them later
    file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    file_paths = [os.path.join(path, f) for f in file_list]
    
    print(f"Loading {len(file_paths)} DICOM files in parallel...")
    max_workers = min(cpu_count() * 2, len(file_paths))
    
    dicoms = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(load_single_dicom_file, fp): fp for fp in file_paths}
        
        for future in as_completed(future_to_path):
            result = future.result()
            if result is not None:
                dicoms.append(result)
    
    print(f"Successfully loaded {len(dicoms)} DICOM files")

    # TASK: create a series_for_inference variable that will contain a list of only 
    # those PyDicom objects that represent files that belong to the series that you 
    # will run inference on.
    # It is important to note that radiological modalities most often operate in terms
    # of studies, and it will most likely be on you to establish criteria for figuring 
    # out which one of the multiple series sent by the scanner is the one you need to feed to 
    # your algorithm. In our case it's rather easy - we have reached an agreement with 
    # people who configured the HippoCrop tool and they label the output of their tool in a 
    # certain way. Can you figure out which is that? 
    # Hint: inspect the metadata of HippoCrop series
    series_for_inference = [dcm for dcm in dicoms if "HippoCrop" in dcm.SeriesDescription]

    if len(series_for_inference) == 0:
        print(f"No series with 'HippoCrop' in SeriesDescription. Using all {len(dicoms)} files from directory.")
        series_for_inference = dicoms

    # Check if there are more than one series (using set comprehension).
    if len(series_for_inference) > 0 and len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        print("Error: Multiple different series found in directory")
        return []

    return series_for_inference

def os_command(command):
    # Comment this if running under Windows
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", command])
    sp.communicate()

    # Uncomment this if running under Windows
    # os.system(command)

if __name__ == "__main__":
    # This code expects a single command line argument with link to the directory containing
    # routed studies
    if len(sys.argv) != 2:
        print("You should supply one command line argument pointing to the routing folder. Exiting.")
        sys.exit()

    # Find all subdirectories within the supplied directory. We assume that 
    # one subdirectory contains a full study
    subdirs = [os.path.join(sys.argv[1], d) for d in os.listdir(sys.argv[1]) if
                os.path.isdir(os.path.join(sys.argv[1], d))]

    hcrop_dirs = [d for d in subdirs if 'HCropVolume' in os.path.basename(d) or 'HippoCrop' in os.path.basename(d)]
    
    if not hcrop_dirs:
        print(f"Error: Could not find HippoCrop volume directory in {sys.argv[1]}")
        print(f"Available directories: {[os.path.basename(d) for d in subdirs]}")
        sys.exit(1)
    
    # Get the latest directory
    study_dir = sorted(hcrop_dirs, key=lambda dir: os.stat(dir).st_mtime, reverse=True)[0]

    print(f"Looking for series to run inference on in directory {study_dir}...")

    # TASK: get_series_for_inference is not complete. Go and complete it
    volume, header = load_dicom_volume_as_numpy_from_list(get_series_for_inference(study_dir))
    print(f"Found series of {volume.shape[2]} axial slices")

    print("HippoVolume.AI: Running inference...")
    # TASK: Use the UNetInferenceAgent class and model parameter file from the previous section
    inference_agent = UNetInferenceAgent(
        device="cpu",
        parameter_file_path=r"../out/2025-12-03_0609_Basic_unet/model.pth")

    # Run inference
    # TASK: single_volume_inference_unpadded takes a volume of arbitrary size 
    # and reshapes y and z dimensions to the patch size used by the model before 
    # running inference. Your job is to implement it.
    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    # TASK: get_predicted_volumes is not complete. Go and complete it
    pred_volumes = get_predicted_volumes(pred_label)

    # Create and save the report
    print("Creating and pushing report...")
    report_save_path = r"/home/workspace/out/report.dcm"
    # TASK: create_report is not complete. Go and complete it. 
    # STAND OUT SUGGESTION: save_report_as_dcm has some suggestions if you want to expand your
    # knowledge of DICOM format
    report_img = create_report(pred_volumes, header, volume, pred_label)
    save_report_as_dcm(header, report_img, report_save_path)

    # Send report to our storage archive
    # TASK: Write a command line string that will issue a DICOM C-STORE request to send our report
    # to our Orthanc server (that runs on port 4242 of the local machine), using storescu tool
    os_command("storescu 127.0.0.1 4242 -v -aec HIPPOAI " + report_save_path)

    # This line will remove the study dir if run as root user
    # Sleep to let our StoreSCP server process the report (remember - in our setup
    # the main archive is routing everyting that is sent to it, including our freshly generated
    # report) - we want to give it time to save before cleaning it up
    time.sleep(2)
    shutil.rmtree(study_dir, onerror=lambda f, p, e: print(f"Error deleting: {e[1]}"))

    print(f"Inference successful on {header['SOPInstanceUID'].value}, out: {pred_label.shape}",
          f"volume ant: {pred_volumes['anterior']}, ",
          f"volume post: {pred_volumes['posterior']}, total volume: {pred_volumes['total']}")
