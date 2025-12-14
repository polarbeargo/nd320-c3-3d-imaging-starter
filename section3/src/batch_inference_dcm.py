"""
Batch Study Processing for HippoVolume.AI

This script processes multiple patient studies in parallel for efficient clinical workflow.
Designed for high-throughput scenarios where multiple studies need inference.

Features:
    - Parallel study processing using ProcessPoolExecutor
    - Robust error handling with per-study success/failure tracking
    - Batch report generation with aggregate statistics
    - Progress monitoring and performance metrics
    - Configurable parallelism based on available CPU cores
    - Optional DICOM C-STORE integration with Orthanc PACS server

Usage:
    python batch_inference_dcm.py /path/to/studies/directory [--max-workers N] [--output-dir PATH] [--send-to-orthanc]
    
Examples:
    # Basic batch processing
    python batch_inference_dcm.py /data/incoming_studies --max-workers 4 --output-dir /data/reports
    
    # With Orthanc PACS integration
    python batch_inference_dcm.py /data/incoming_studies --send-to-orthanc --orthanc-host 127.0.0.1
"""

import os
import sys
import time
import argparse
import json
import subprocess
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pydicom

from inference_dcm import (
    get_series_for_inference,
    load_dicom_volume_as_numpy_from_list,
    get_predicted_volumes,
    create_report,
    save_report_as_dcm,
)
from inference.UNetInferenceAgent import UNetInferenceAgent


def os_command(command):
    """Execute a shell command
    
    Arguments:
        command: Shell command string to execute
    """
    # Use non-interactive shell and redirect output to suppress verbose storescu output
    # Comment this if running under Windows
    sp = subprocess.Popen(
        ["/bin/bash", "-c", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL
    )
    sp.communicate()

    # Uncomment this if running under Windows
    # os.system(command)


class BatchProcessingStats:
    """Tracks statistics for batch processing"""
    
    def __init__(self):
        self.total_studies = 0
        self.successful = 0
        self.failed = 0
        self.start_time = None
        self.end_time = None
        self.results = []
        
    def start(self):
        self.start_time = time.time()
        
    def finish(self):
        self.end_time = time.time()
        
    def add_success(self, study_name: str, result: Dict):
        self.successful += 1
        self.results.append({
            'study': study_name,
            'status': 'success',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    def add_failure(self, study_name: str, error: str):
        self.failed += 1
        self.results.append({
            'study': study_name,
            'status': 'failed',
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
        
    def get_elapsed_time(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
        
    def get_summary(self) -> Dict:
        elapsed = self.get_elapsed_time()
        return {
            'total_studies': self.total_studies,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate': f"{(self.successful/self.total_studies*100):.1f}%" if self.total_studies > 0 else "0%",
            'elapsed_time_seconds': elapsed,
            'elapsed_time_minutes': elapsed / 60,
            'avg_time_per_study': elapsed / self.total_studies if self.total_studies > 0 else 0,
            'results': self.results
        }


def find_hippocrop_studies(root_dir: str) -> List[str]:
    """
    Recursively find all HippoCrop study directories
    
    Arguments:
        root_dir: Root directory to search for studies
        
    Returns:
        List of full paths to HippoCrop directories
    """
    hippocrop_dirs = []
    
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory")
        return []
    
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    if 'HippoCrop' in subitem or 'HCropVolume' in subitem:
                        hippocrop_dirs.append(subitem_path)
    
    return sorted(hippocrop_dirs)


def validate_study_directory(study_dir: str) -> Tuple[bool, str]:
    """
    Validate that a study directory contains valid DICOM files
    
    Arguments:
        study_dir: Path to study directory
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not os.path.isdir(study_dir):
        return False, "Not a directory"
    
    files = [f for f in os.listdir(study_dir) if os.path.isfile(os.path.join(study_dir, f))]
    
    if len(files) == 0:
        return False, "No files found"
    
    dicom_count = 0
    for f in files[:5]:
        try:
            pydicom.dcmread(os.path.join(study_dir, f))
            dicom_count += 1
        except:
            continue
    
    if dicom_count == 0:
        return False, "No valid DICOM files found"
    
    return True, f"Valid ({len(files)} files, {dicom_count} DICOM confirmed)"


def process_single_study(args: Tuple) -> Dict:
    """
    Process a single study (designed to be called in parallel)
    
    Arguments:
        args: Tuple of (study_dir, model_path, output_dir, study_index, total_studies, send_to_orthanc, orthanc_host, orthanc_port, orthanc_aec)
        
    Returns:
        Dictionary with processing results or error information
    """
    study_dir, model_path, output_dir, study_index, total_studies, send_to_orthanc, orthanc_host, orthanc_port, orthanc_aec = args
    study_name = os.path.basename(study_dir)
    
    result = {
        'study_name': study_name,
        'study_dir': study_dir,
        'success': False,
        'error': None,
        'volumes': None,
        'num_slices': 0,
        'processing_time': 0
    }
    
    start_time = time.time()
    
    try:
        print(f"[{study_index}/{total_studies}] Processing {study_name}...")
        
        is_valid, msg = validate_study_directory(study_dir)
        if not is_valid:
            raise ValueError(f"Invalid study directory: {msg}")
        
        series = get_series_for_inference(study_dir)
        if len(series) == 0:
            raise ValueError("No valid series found for inference")
        
        volume, header = load_dicom_volume_as_numpy_from_list(series)
        result['num_slices'] = volume.shape[2]
        print(f"  → Loaded {volume.shape[2]} slices")
        
        inference_agent = UNetInferenceAgent(
            device="cpu",
            parameter_file_path=model_path
        )
        pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
        pred_volumes = get_predicted_volumes(pred_label)
        result['volumes'] = pred_volumes
        
        print(f"  → Anterior: {pred_volumes['anterior']}, "
              f"Posterior: {pred_volumes['posterior']}, "
              f"Total: {pred_volumes['total']} voxels")
        
        report_img = create_report(pred_volumes, header, volume, pred_label)
        report_filename = f"report_{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dcm"
        report_path = os.path.join(output_dir, report_filename)
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.abspath(report_path)
        
        save_report_as_dcm(header, report_img, report_path)
        
        result['report_path'] = report_path
        print(f"  → Report saved: {report_path}")

        if not os.path.isfile(report_path):
            raise ValueError(f"Report file was not created: {report_path}")
        
        if send_to_orthanc:
            try:
                print(f"  → Sending report to Orthanc at {orthanc_host}:{orthanc_port}...")
                if not os.path.isfile(report_path):
                    raise FileNotFoundError(f"Report file not found: {report_path}")
                
                result_code = subprocess.run(
                    ["storescu", orthanc_host, str(orthanc_port), "-aec", orthanc_aec, report_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    timeout=30
                )
                
                if result_code.returncode == 0:
                    result['sent_to_orthanc'] = True
                    print(f"  ✓ Report sent to Orthanc successfully")
                else:
                    error_msg = result_code.stderr.decode() if result_code.stderr else "Unknown error"
                    raise RuntimeError(f"storescu failed with code {result_code.returncode}: {error_msg}")
                    
            except subprocess.TimeoutExpired:
                result['sent_to_orthanc'] = False
                result['orthanc_error'] = "Timeout waiting for Orthanc response"
                print(f"  ⚠ Warning: Orthanc upload timed out after 30 seconds")
            except FileNotFoundError as e:
                if 'storescu' in str(e):
                    result['sent_to_orthanc'] = False
                    result['orthanc_error'] = "storescu command not found - install DCMTK"
                    print(f"  ⚠ Warning: storescu not installed. Install DCMTK to enable Orthanc integration")
                else:
                    raise
            except Exception as orthanc_error:
                result['sent_to_orthanc'] = False
                result['orthanc_error'] = str(orthanc_error)
                print(f"  ⚠ Warning: Failed to send to Orthanc: {orthanc_error}")
        
        result['success'] = True
        
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        print(f"  ✓ Completed in {processing_time:.2f}s")
        
    except Exception as e:
        result['error'] = str(e)
        result['processing_time'] = time.time() - start_time
        print(f"  ✗ Failed: {e}")
    
    return result


def run_batch_inference(
    studies_root: str,
    model_path: str,
    output_dir: str,
    max_workers: Optional[int] = None,
    send_to_orthanc: bool = False,
    orthanc_host: str = "127.0.0.1",
    orthanc_port: int = 4242,
    orthanc_aec: str = "HIPPOAI"
) -> BatchProcessingStats:
    """
    Run inference on multiple studies in parallel
    
    Arguments:
        studies_root: Root directory containing study folders
        model_path: Path to trained model file
        output_dir: Directory to save reports
        max_workers: Maximum parallel workers (None = auto-detect)
        send_to_orthanc: Whether to send reports to Orthanc PACS
        orthanc_host: Orthanc server hostname/IP
        orthanc_port: Orthanc DICOM port
        orthanc_aec: Application Entity Title for Orthanc
        
    Returns:
        BatchProcessingStats object with results
    """
    stats = BatchProcessingStats()
    stats.start()
    
    print(f"\n{'='*70}")
    print(f"HippoVolume.AI - Batch Study Processing")
    print(f"{'='*70}\n")
    print(f"Searching for studies in: {studies_root}")
    
    study_dirs = find_hippocrop_studies(studies_root)
    
    if len(study_dirs) == 0:
        print("ERROR: No HippoCrop study directories found!")
        print(f"Searched in: {studies_root}")
        stats.finish()
        return stats
    
    stats.total_studies = len(study_dirs)
    print(f"Found {len(study_dirs)} studies to process\n")
    
    if max_workers is None:
        max_workers = min(cpu_count(), len(study_dirs))
    else:
        max_workers = min(max_workers, len(study_dirs))
    
    print(f"Configuration:")
    print(f"  - Parallel workers: {max_workers}")
    print(f"  - Model: {model_path}")
    print(f"  - Output directory: {output_dir}")
    if send_to_orthanc:
        print(f"  - Send to Orthanc: YES ({orthanc_host}:{orthanc_port}, AEC: {orthanc_aec})")
    else:
        print(f"  - Send to Orthanc: NO")
    print(f"\nStarting batch processing...\n")
    
    process_args = [
        (study_dir, model_path, output_dir, idx + 1, len(study_dirs), 
         send_to_orthanc, orthanc_host, orthanc_port, orthanc_aec)
        for idx, study_dir in enumerate(study_dirs)
    ]
 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_study, args): args[0] for args in process_args}
        
        for future in as_completed(futures):
            study_dir = futures[future]
            study_name = os.path.basename(study_dir)
            
            try:
                result = future.result()
                if result['success']:
                    stats.add_success(study_name, result)
                else:
                    stats.add_failure(study_name, result['error'])
            except Exception as e:
                stats.add_failure(study_name, str(e))
    
    stats.finish()
    return stats


def save_batch_summary(stats: BatchProcessingStats, output_path: str):
    """
    Save batch processing summary to JSON file
    
    Arguments:
        stats: BatchProcessingStats object
        output_path: Path to save JSON summary
    """
    summary = stats.get_summary()
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBatch summary saved to: {output_path}")


def print_batch_summary(stats: BatchProcessingStats):
    """
    Print formatted batch processing summary
    
    Arguments:
        stats: BatchProcessingStats object
    """
    summary = stats.get_summary()
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*70}\n")
    print(f"Total Studies:     {summary['total_studies']}")
    print(f"Successful:        {summary['successful']} ✓")
    print(f"Failed:            {summary['failed']} ✗")
    print(f"Success Rate:      {summary['success_rate']}")
    print(f"\nPerformance:")
    print(f"Total Time:        {summary['elapsed_time_minutes']:.2f} minutes")
    print(f"Avg Time/Study:    {summary['avg_time_per_study']:.2f} seconds")
    
    if summary['failed'] > 0:
        print(f"\nFailed Studies:")
        for result in summary['results']:
            if result['status'] == 'failed':
                print(f"  - {result['study']}: {result['error']}")
    
    print(f"\n{'='*70}\n")


def main():
    """Main entry point for batch processing"""
    parser = argparse.ArgumentParser(
        description='Batch processing for HippoVolume.AI clinical inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all studies in directory with auto-detected workers
  python batch_inference_dcm.py /data/TestVolumes
  
  # Process with 4 parallel workers
  python batch_inference_dcm.py /data/TestVolumes --max-workers 4
  
  # Specify custom output directory
  python batch_inference_dcm.py /data/TestVolumes --output-dir /data/reports
  
  # Process and send reports to Orthanc PACS server
  python batch_inference_dcm.py /data/TestVolumes --send-to-orthanc
  
  # Send to Orthanc with custom host and port
  python batch_inference_dcm.py /data/TestVolumes --send-to-orthanc --orthanc-host 192.168.1.100 --orthanc-port 4242
        """
    )
    
    parser.add_argument(
        'studies_root',
        type=str,
        help='Root directory containing study folders (e.g., Study1, Study2, etc.)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='../out/2025-12-03_0609_Basic_unet/model.pth',
        help='Path to trained model file (default: ../out/2025-12-03_0609_Basic_unet/model.pth)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../out/batch_reports',
        help='Directory to save batch reports (default: ../out/batch_reports)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of parallel workers (default: auto-detect based on CPU cores)'
    )
    
    parser.add_argument(
        '--save-summary',
        action='store_true',
        help='Save detailed JSON summary of batch processing'
    )
    
    parser.add_argument(
        '--send-to-orthanc',
        action='store_true',
        help='Send generated reports to Orthanc PACS server via DICOM C-STORE'
    )
    
    parser.add_argument(
        '--orthanc-host',
        type=str,
        default='127.0.0.1',
        help='Orthanc server hostname or IP address (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--orthanc-port',
        type=int,
        default=4242,
        help='Orthanc DICOM port (default: 4242)'
    )
    
    parser.add_argument(
        '--orthanc-aec',
        type=str,
        default='HIPPOAI',
        help='Application Entity Title for Orthanc (default: HIPPOAI)'
    )
    
    args = parser.parse_args()
    studies_root = os.path.abspath(args.studies_root)
    model_path = os.path.abspath(args.model_path)
    output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.isdir(studies_root):
        print(f"ERROR: Studies root directory does not exist: {studies_root}")
        sys.exit(1)
    
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file does not exist: {model_path}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    stats = run_batch_inference(
        studies_root=studies_root,
        model_path=model_path,
        output_dir=output_dir,
        max_workers=args.max_workers,
        send_to_orthanc=args.send_to_orthanc,
        orthanc_host=args.orthanc_host,
        orthanc_port=args.orthanc_port,
        orthanc_aec=args.orthanc_aec
    )
    
    print_batch_summary(stats)
    if args.save_summary:
        summary_path = os.path.join(
            args.output_dir,
            f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        save_batch_summary(stats, summary_path)
    
    sys.exit(0 if stats.failed == 0 else 1)


if __name__ == "__main__":
    main()
