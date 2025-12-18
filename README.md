## Quantifying Alzheimer's Disease Progression Through Automated Measurement of Hippocampal Volume

Alzheimer's disease (AD) is a progressive neurodegenerative disorder that results in impaired neuronal (brain cell) function and eventually, cell death. AD is the most common cause of dementia. Clinically, it is characterized by memory loss, inability to learn new material, loss of language function, and other manifestations. 

For patients exhibiting early symptoms, quantifying disease progression over time can help direct therapy and disease management. 

A radiological study via MRI exam is currently one of the most advanced methods to quantify the disease. In particular, the measurement of hippocampal volume has proven useful to diagnose and track progression in several brain disorders, most notably in AD. Studies have shown reduced volume of the hippocampus in patients with AD.

The hippocampus is a critical structure of the human brain (and the brain of other vertebrates) that plays important roles in the consolidation of information from short-term memory to long-term memory. In other words, the hippocampus is thought to be responsible for memory and learning (that's why we are all here, after all!)

![Hippocampus](./readme.img/Hippocampus_small.gif)

Humans have two hippocampi, one in each hemishpere of the brain. They are located in the medial temporal lobe of the brain. Fun fact - the word "hippocampus" is roughly translated from Greek as "horselike" because of the similarity to a seahorse, a peculiarity observed by one of the first anatomists to illustrate the structure.

<img src="./readme.img/Hippocampus_and_seahorse_cropped.jpg" width=200/>

According to [studies](https://www.sciencedirect.com/science/article/pii/S2213158219302542), the volume of the hippocampus varies in a population, depending on various parameters, within certain boundaries, and it is possible to identify a "normal" range when taking into account age, sex and brain hemisphere. 

<img src="./readme.img/nomogram_fem_right.svg" width=300>

There is one problem with measuring the volume of the hippocampus using MRI scans, though - namely, the process tends to be quite tedious since every slice of the 3D volume needs to be analyzed, and the shape of the structure needs to be traced. The fact that the hippocampus has a non-uniform shape only makes it more challenging. Do you think you could spot the hippocampi in this axial slice?

<img src="./readme.img/mri.jpg" width=200>

As you might have guessed by now, we are going to build a piece of AI software that could help clinicians perform this task faster and more consistently.

You have seen throughout the course that a large part of AI development effort is taken up by curating the dataset and proving clinical efficacy. In this project, we will focus on the technical aspects of building a segmentation model and integrating it into the clinician's workflow, leaving the dataset curation and model validation questions largely outside the scope of this project. You will build an end-to-end AI system which features a machine learning algorithm that integrates into a clinical-grade viewer and automatically measures hippocampal volumes of new patients, as their studies are committed to the clinical imaging archive.


## The Dataset

We are using the "Hippocampus" dataset from the [Medical Decathlon competition](http://medicaldecathlon.com/). This [dataset](https://github.com/udacity/nd320-c3-3d-imaging-starter/tree/master/data/TrainingSet) is stored as a collection of NIFTI files, with one file per volume, and one file per corresponding segmentation mask. The original images here are T2 MRI scans of the full brain. As noted, in this dataset we are using cropped volumes where only the region around the hippocampus has been cut out. This makes the size of our dataset quite a bit smaller, our machine learning problem a bit simpler and allows us to have reasonable training times. You should not think of it as "toy" problem, though. Algorithms that crop rectangular regions of interest are quite common in medical imaging. Segmentation is still hard.

## The Programming Environment

You will have two options for the environment to use throughout this project:

### Udacity Workspaces

#### Running EDA in Udacity Workspace for Section 1

**Step 1: Launch Jupyter Notebook**

Open and execute the EDA notebook to perform data exploration and cleaning:

```bash
cd /home/workspace/
jupyter notebook --port 3002 --ip=0.0.0.0 --allow-root
```

**Step 2: Execute All Cells**

Run all cells in the notebook to:
- Explore the hippocampus dataset
- Analyze volume distributions and statistics
- Clean and validate the data
- Generate the cleaned dataset in the `out` folder

**Step 3: Download the out folder**

After successful execution, download the `out` folder containing the cleaned dataset. This folder will be used in Section 2 for model training.

#### Running the ML Pipeline in Udacity Workspace for Section 2

**Step 1: Upload the out folder**

Upload the `out` folder (containing the clean dataset from Section 1) to the section 2 workspace.

**Step 2: Run the ML Training Pipeline**

```bash
cd src
python run_ml_pipeline.py
```

**Step 3: Launch TensorBoard**

To monitor training progress in real-time, launch TensorBoard from the same directory:

```bash
tensorboard --logdir runs --bind_all
```

Then access TensorBoard in your browser at the provided URL.

#### Running the System in Udacity Workspace for section 3

The Udacity workspace is pre-configured with all necessary tools. To run the complete inference pipeline with PACS integration:  

**Step 1: Launch Services (in separate terminals)**

```bash
# Terminal 1 - Start Orthanc PACS server
./launch_orthanc.sh

# Terminal 2 - Start OHIF viewer
./launch_OHIF.sh
```

**Step 2: Install DICOM Tools (first time only)**

```bash
apt-get install dcmtk
```

**Step 3: Test DICOM Routing**

```bash
./deploy_scripts/send_volume.sh
```

**Step 4: Run Inference**

Option A - Single study inference:
```bash
python inference_dcm.py /data/TestVolumes/Study1
```

Option B - Batch processing (multiple studies in parallel):
```bash
python batch_inference_dcm.py /data/TestVolumes --send-to-orthanc
```

The reports will be automatically sent to Orthanc PACS and viewable in the OHIF viewer.

### Local Environment

If you would like to run the project locally, you would need a Python 3.7+ environment with the following libraries for the first two sections of the project:

* nibabel
* matplotlib
* numpy
* pydicom
* PIL
* json
* torch (preferably with CUDA)
* tensorboard

In the 3rd section of the project we will be working with three software products for emulating the clinical network. You would need to install and configure:

* [Orthanc server](https://www.orthanc-server.com/download.php) for PACS emulation
* [OHIF zero-footprint web viewer](https://docs.ohif.org/development/getting-started.html) for viewing images. Note that if you deploy OHIF from its github repository, at the moment of writing the repo includes a yarn script (`orthanc:up`) where it downloads and runs the Orthanc server from a Docker container. If that works for you, you won't need to install Orthanc separately.
* If you are using Orthanc (or other DICOMWeb server), you will need to configure OHIF to read data from your server. OHIF has instructions for this: https://docs.ohif.org/configuring/data-source.html
* In order to fully emulate the Udacity workspace, you will also need to configure Orthanc for auto-routing of studies to automatically direct them to your AI algorithm. For this you will need to take the script that you can find at `section3/src/deploy_scripts/route_dicoms.lua` and install it to Orthanc as explained on this page: https://book.orthanc-server.com/users/lua.html
* [DCMTK tools](https://dcmtk.org/) for testing and emulating a modality. Note that if you are running a Linux distribution, you might be able to install dcmtk directly from the package manager (e.g. `apt-get install dcmtk` in Ubuntu)


**Workspace 3**. This workspace is a simpler hardware, with no GPU, which is more representative of a clinical environment. This workspace also has a few tools installed in it, which is replicates the following clinical network setup:  
<img src="./readme.img/network_setup.png" width=400em>

**Resource might help:** For a detailed guide on clinical workflow integration with Orthanc and OHIF Viewer, see [this article](https://medium.com/@GaganaB/clinical-workflow-integration-with-orthanc-and-ohif-viewer-38930a3e94de).

Specifically, we have the following software in this setup:

* MRI scanner is represented by a script `section3/src/deploy_scripts/send_volume.sh`. When you run this script it will simulate what happens after a radiological exam is complete, and send a volume to the clinical PACS. Note that scanners typically send entire studies to archives.
* PACS server is represented by [Orthanc](http://orthanc-server.com/) deployment that is listening to DICOM DIMSE requests on port 4242. Orthanc also has a DicomWeb interface that is exposed at port 8042, prefix /dicom-web. There is no authentication and you are welcome to explore either one of the mechanisms of access using a tool like curl or Postman. Our PACS server is also running an auto-routing module that sends a copy of everything it receives to an AI server. See instructions ad the end of this page on how to launch if you are using the Udacity Workspace.  
* Viewer system is represented by [OHIF](http://ohif.org/). It is connecting to the Orthanc server using DicomWeb and is serving a web application on port 3000. Again, see instructions at the end of this page if you are using the Udacity Workspace.
* AI server is represented by a couple of scripts. `section3/src/deploy_scripts/start_listener.sh` brings up a DCMTK's `storescp` and configures it to just copy everything it receives into a directory that you will need to specify by editing this script, organizing studies as one folder per study. HippoVolume.AI is the AI module that you will create in this section.  

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Deep Learning** | PyTorch 1.4 | Model training & inference |
| **Architecture** | 2D UNet | Segmentation network |
| **Medical I/O** | MedPy, PyDicom | NIFTI & DICOM handling |
| **Parallelization** | ThreadPoolExecutor, ProcessPoolExecutor | I/O & compute optimization |
| **Visualization** | TensorBoard, PIL | Training monitoring & reports |
| **PACS Integration** | Orthanc, DCMTK | Clinical deployment |
| **Data Science** | NumPy, SciPy | Array operations |
---

## System Flow Diagram

```mermaid
flowchart TD
    Start([MRI Scans]) --> EDA[Section 1: EDA<br/>Exploratory Analysis]
    EDA --> Clean[(Clean Dataset<br/>260 volumes)]
    
    Clean --> Train[Section 2: Training]
    Train --> Load[HippocampusDatasetLoader<br/>ProcessPoolExecutor]
    Load --> Slice[SlicesDataset<br/>2D Slices]
    Slice --> DataLoader[PyTorch DataLoader<br/>Batch=8, Workers=4]
    DataLoader --> UNet[2D UNet Model<br/>3 Classes]
    UNet --> Optimize[Training Loop<br/>Adam + AMP]
    Optimize --> Val[Validation]
    Val --> Model[(Trained Model<br/>Dice: 0.8928)]
    
    Model --> Deploy[Section 3: Deployment]
    PACS[PACS/Orthanc] --> Router[DICOM Router]
    Router --> Incoming["File System<br/>/data/incoming"]
    Incoming --> InfScript[inference_dcm.py]
    
    InfScript --> LoadDCM[Load DICOM<br/>ThreadPoolExecutor]
    LoadDCM --> Construct[Construct 3D Volume]
    Construct --> Agent[UNetInferenceAgent]
    Agent --> Model
    Agent --> Predict[Predictions]
    Predict --> Volumes[Compute Volumes<br/>Anterior + Posterior]
    Volumes --> Report[Generate Report<br/>ThreadPoolExecutor]
    Report --> SendPACS[Send to PACS<br/>storescu]
    SendPACS --> PACS
    
    style EDA fill:#E3F2FD
    style Train fill:#C8E6C9
    style Deploy fill:#FFF9C4
    style Model fill:#4CAF50,color:#fff
    style PACS fill:#FF5722,color:#fff
```

## Model Performance

The baseline **2D UNet model** achieves excellent performance on hippocampus segmentation:

| Metric | Score |
|--------|-------|
| **Mean Dice Coefficient** | **0.8928** |
| Architecture | 2D UNet (Recursive) |
| Classes | 3 (Background, Anterior, Posterior) |
| Patch Size | 64×64 |
| Training Epochs | 10 |
| Optimizer | Adam (lr=0.0002) |

This performance demonstrates the model is **production-ready** for clinical deployment. The Dice score of [0.8928](section2/out/2025-12-03_0609_Basic_unet/results.json) indicates strong overlap between predicted and ground truth hippocampal segmentations, which is considered excellent for medical image segmentation tasks.

**Key Optimizations:**
- Parallel data loading with ProcessPoolExecutor
- GPU acceleration with Automatic Mixed Precision (AMP)
- Batched inference for efficiency
- Parallel DICOM processing in deployment pipeline
- Batch study processing for multi-study throughput (2-8× parallel speedup, 10-20× vs. baseline clinical workflow)

> **Note:** Experimental features like Test-Time Augmentation (TTA) and 3D morphological post-processing were tested but removed as they decreased performance. The baseline 2D UNet without augmentation provides the best results.

| Optimization | Speedup | Impact Area | File(s) Modified |
|--------------|---------|-------------|------------------|
| Parallel Data Loading | 3-8x | Data loading, training | `HippocampusDatasetLoader.py`, `UNetExperiment.py` |
| Mixed Precision Training (AMP) | 2-3x | Training | `UNetExperiment.py`, `run_ml_pipeline.py` |
| Batched Inference | 5-10x | Testing, deployment | `inference/UNetInferenceAgent.py` |
| Parallel DICOM Loading | 3-5x | Study ingestion (single study) | `inference_dcm.py` |
| Parallel Report Generation | 2-3x | Report creation (single study) | `inference_dcm.py` |
| **Batch Study Processing** | **N×** | **Multi-study throughput** | `batch_inference_dcm.py` |
| Multi-Process Parallelism | 2-8x | Concurrent study processing | `batch_inference_dcm.py` (ProcessPoolExecutor) |
| Hierarchical Parallelization | 1.5-3x | Combined study + I/O parallelism | `batch_inference_dcm.py` (Process + Thread pools) |

**Combined Effect**: 
- Training time reduced by ~5-10x
- Single study inference time reduced by 5-10x
- **Batch processing throughput: N studies can be processed in ~N/max_workers time** (e.g., 10 studies with 4 workers ≈ 2.5× single study time)
- Clinical workflow time reduced by ~5-8x for typical studies, **10-20x for batch operations**

---

## TensorBoard — Images

<img src="./section2/out/tensorboard_images.png" width="600"/>

## TensorBoard — Scalars

<img src="./section2/out/tensorboard_scalars.png" width="600"/>


## OHIF Viewer displaying a study and segmentation results
<img src="./section3/out/OHIFViewer2.png" width="600"/>
<img src="./section3/out/OHIFViewer.png" width="600"/>

## Parallelization Strategy

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#4CAF50', 'primaryTextColor':'#fff', 'primaryBorderColor':'#2E7D32', 'secondaryColor':'#2196F3', 'secondaryTextColor':'#fff', 'secondaryBorderColor':'#1565C0', 'tertiaryColor':'#FF9800', 'tertiaryTextColor':'#fff', 'tertiaryBorderColor':'#E65100', 'lineColor':'#9E9E9E'}}}%%
mindmap
  root((Parallelization))
    Training
      ProcessPoolExecutor
        NIFTI Loading
        Volume Processing
      DataLoader Workers
        Batch Prefetch
        num_workers=4
      GPU Acceleration
        CUDA
        Mixed Precision AMP
    Inference
      ThreadPoolExecutor
        DICOM I/O
        Report Slices
      Batch Processing
        16 slices/batch
        CPU inference
    Future
      Multi-Study
        Parallel Studies
        Queue Management
```
| Task | Location | Executor Type | Reason |
|------|----------|---------------|---------|
| **Volume calculation** | Section 1 | ProcessPoolExecutor | CPU-bound (numpy, decompression) |
| **File copying** | Section 1 | ProcessPoolExecutor | CPU-bound (large file I/O + verification) |
| **NIFTI loading** | Section 2 | ProcessPoolExecutor | Mixed (I/O + heavy CPU from .gz decompression) |
| **DICOM loading** | Section 3 (inference_dcm.py) | ThreadPoolExecutor | I/O-bound (simple file reading) |
| **Report generation** | Section 3 (inference_dcm.py) | ThreadPoolExecutor | GIL-released (PIL/numpy operations) |
| **Batch study processing** | Section 3 (batch_inference_dcm.py) | ProcessPoolExecutor | Study-level parallelism (bypasses Python GIL, independent processes) |
| **Per-study DICOM loading** | Section 3 (batch_inference_dcm.py) | ThreadPoolExecutor | I/O-bound (inherited from inference_dcm.py functions) |
| **Per-study report generation** | Section 3 (batch_inference_dcm.py) | ThreadPoolExecutor | GIL-released (inherited from inference_dcm.py functions) |

---

## Performance Metrics

```mermaid
graph LR
    subgraph "Model Performance"
        A[Training Dice<br/>0.8928]
        B[Validation<br/>Monitored]
        C[Test Set<br/>Evaluated]
    end
    
    subgraph "System Performance"
        D[Data Loading<br/>Parallel]
        E[GPU Training<br/>AMP Enabled]
        F[Inference<br/>Batched]
    end
    
    subgraph "Clinical Metrics"
        G[Volume Anterior<br/>~1900 voxels]
        H[Volume Posterior<br/>~1450 voxels]
        I[Total Volume<br/>~3370 voxels]
    end
    
    style A fill:#4CAF50,color:#fff
    style E fill:#2196F3,color:#fff
    style I fill:#FF9800,color:#fff
```

## 2. Class Diagram - Section 2 (Training)

The `Config` class holds hyperparameters that configure the `UNetExperiment`, which orchestrates the entire training workflow. The experiment uses `HippocampusDatasetLoader` with `ProcessPoolExecutor` for parallel NIFTI loading, feeds data through `SlicesDataset` to PyTorch's `DataLoader`, and trains the recursive `UNet` model composed of `UnetSkipConnectionBlock` layers. The `UNetInferenceAgent` performs test-time inference, while `VolumeStats` computes Dice and Jaccard metrics for model evaluation.

```mermaid
classDiagram
    class Config {
        +str name
        +str root_dir
        +int n_epochs
        +int batch_size
        +int patch_size
        +float learning_rate
        +str test_results_dir
        +bool use_amp
    }
    
    class UNetExperiment {
        -int n_epochs
        -dict split
        -str out_dir
        -int epoch
        -str name
        -DataLoader train_loader
        -DataLoader val_loader
        -list test_data
        -device device
        -UNet model
        -loss_function loss_function
        -Optimizer optimizer
        -Scheduler scheduler
        -bool use_amp
        -GradScaler scaler
        -SummaryWriter tensorboard
        +__init__(config, split, dataset)
        +train()
        +validate()
        +run_test()
        +run()
        +save_model_parameters(path)
        +load_model_parameters(path)
    }
    
    class UNet {
        -UnetSkipConnectionBlock model
        -int num_classes
        -int in_channels
        -int initial_filter_size
        +__init__(num_classes, in_channels, initial_filter_size, kernel_size, num_downs, norm_layer)
        +forward(x) Tensor
    }
    
    class UnetSkipConnectionBlock {
        -bool outermost
        -Sequential model
        -nn.MaxPool2d pool
        -nn.Conv2d conv1
        -nn.Conv2d conv2
        -nn.ConvTranspose2d upconv
        +__init__(in_channels, out_channels, num_classes, kernel_size, submodule, outermost, innermost, norm_layer, use_dropout)
        +forward(x) Tensor
        +contract(in_channels, out_channels, kernel_size, norm_layer) Sequential
        +expand(in_channels, out_channels, kernel_size) Sequential
    }
    
    class SlicesDataset {
        -list data
        -list slices
        +__init__(data)
        +__getitem__(idx) dict
        +__len__() int
    }
    
    class HippocampusDatasetLoader {
        +LoadHippocampusData(root_dir, y_shape, z_shape) ndarray
        -process_single_file(args) dict
    }
    
    class UNetInferenceAgent {
        -UNet model
        -int patch_size
        -device device
        +__init__(parameter_file_path, model, device, patch_size)
        +single_volume_inference(volume) ndarray
        +single_volume_inference_unpadded(volume) ndarray
    }
    
    class VolumeStats {
        +Dice3d(a, b) float
        +Jaccard3d(a, b) float
    }
    
    class Utils {
        +med_reshape(image, new_shape, pad_value) ndarray
        +log_to_tensorboard(writer, loss, data, target, prediction, prediction_softmax, counter)
    }
    
    Config --> UNetExperiment : configures
    UNetExperiment --> UNet : creates/trains
    UNetExperiment --> SlicesDataset : uses
    UNetExperiment --> UNetInferenceAgent : uses for testing
    UNetExperiment --> VolumeStats : evaluates with
    UNet --> UnetSkipConnectionBlock : composed of
    UnetSkipConnectionBlock --> UnetSkipConnectionBlock : recursive
    SlicesDataset --> HippocampusDatasetLoader : loads from
    UNetInferenceAgent --> UNet : uses
    UNetInferenceAgent --> Utils : uses
```

---

## 3. Class Diagram - Section 3 (Deployment)

```mermaid
classDiagram
    class inference_dcm {
        +load_dicom_volume_as_numpy_from_list(dcmlist) tuple
        +get_predicted_volumes(pred) dict
        +create_report(inference, header, orig_vol, pred_vol) PIL.Image
        +save_report_as_dcm(header, report, path)
        +get_series_for_inference(path) list
        +process_single_dicom(args) PyDicom
        +process_single_slice_for_report(args) PIL.Image
        +os_command(command)
        +main()
    }
    
    class UNetInferenceAgent_Section3 {
        -UNet model
        -int patch_size
        -device device
        +__init__(parameter_file_path, model, device, patch_size)
        +single_volume_inference(volume) ndarray
        +single_volume_inference_unpadded(volume) ndarray
    }
    
    class UNet_Section3 {
        -UnetSkipConnectionBlock model
        +__init__(num_classes, in_channels, initial_filter_size, kernel_size, num_downs, norm_layer)
        +forward(x) Tensor
    }
    
    class ThreadPoolExecutor {
        <<external>>
        +submit(fn, *args) Future
        +map(fn, *iterables) Iterator
    }
    
    class PyDicom {
        <<external>>
        +dcmread(path) Dataset
        +pixel_array
        +SeriesDescription
        +SeriesInstanceUID
    }
    
    class PIL_Image {
        <<external>>
        +new(mode, size, color) Image
        +fromarray(arr) Image
        +paste(image, box)
    }
    
    class Orthanc {
        <<external PACS>>
        +store(dcm_file)
        +route(study)
    }
    
    inference_dcm --> UNetInferenceAgent_Section3 : uses
    inference_dcm --> ThreadPoolExecutor : parallelizes with
    inference_dcm --> PyDicom : reads DICOM with
    inference_dcm --> PIL_Image : creates report with
    inference_dcm --> Orthanc : sends to
    UNetInferenceAgent_Section3 --> UNet_Section3 : uses
```

This diagram shows the deployment architecture integrating with clinical PACS systems. The `inference_dcm` script orchestrates the entire workflow: using `ThreadPoolExecutor` for parallel DICOM I/O, `PyDicom` for reading medical images, `UNetInferenceAgent` with the trained `UNet` model for hippocampus segmentation, `PIL_Image` for generating visual reports, and finally sending results to `Orthanc` PACS via DICOM C-STORE protocol for clinical review.

---

## 4. Sequence Diagram - Training Pipeline

```mermaid
sequenceDiagram
    participant User
    participant run_ml_pipeline
    participant Config
    participant Loader as HippocampusDatasetLoader
    participant Experiment as UNetExperiment
    participant Dataset as SlicesDataset
    participant Model as UNet
    participant TensorBoard
    
    User->>run_ml_pipeline: python run_ml_pipeline.py
    run_ml_pipeline->>Config: create config
    Config-->>run_ml_pipeline: config object
    
    run_ml_pipeline->>Loader: LoadHippocampusData(root_dir, y_shape, z_shape)
    Note over Loader: Parallel processing with ProcessPoolExecutor
    Loader->>Loader: process_single_file() for each volume
    Loader-->>run_ml_pipeline: dataset array
    
    run_ml_pipeline->>run_ml_pipeline: create train/val/test split
    
    run_ml_pipeline->>Experiment: UNetExperiment(config, split, dataset)
    Experiment->>Dataset: SlicesDataset(train_data)
    Experiment->>Dataset: SlicesDataset(val_data)
    Experiment->>Model: UNet(num_classes=3)
    Experiment->>TensorBoard: SummaryWriter()
    
    run_ml_pipeline->>Experiment: run()
    
    loop for each epoch
        Experiment->>Experiment: train()
        loop for each batch
            Experiment->>Dataset: __getitem__(idx)
            Dataset-->>Experiment: {image, seg}
            Experiment->>Model: forward(image)
            Model-->>Experiment: predictions
            Experiment->>Experiment: compute loss
            Experiment->>Experiment: backpropagation
            Experiment->>TensorBoard: log metrics
        end
        
        Experiment->>Experiment: validate()
        loop for each validation batch
            Experiment->>Dataset: __getitem__(idx)
            Dataset-->>Experiment: {image, seg}
            Experiment->>Model: forward(image)
            Model-->>Experiment: predictions
            Experiment->>Experiment: compute val_loss
        end
        Experiment->>TensorBoard: log validation metrics
    end
    
    Experiment->>Experiment: save_model_parameters()
    Experiment-->>User: Training complete
```

---

## 5. Sequence Diagram - Inference Pipeline

```mermaid
sequenceDiagram
    participant Experiment as UNetExperiment
    participant Agent as UNetInferenceAgent
    participant Model as UNet
    participant Utils as volume_stats
    
    Experiment->>Experiment: run_test()
    
    loop for each test volume
        Experiment->>Agent: single_volume_inference_unpadded(volume)
        Agent->>Agent: reshape volume to patch_size
        
        loop for each slice batch
            Agent->>Agent: prepare batch [batch, 1, H, W]
            Agent->>Model: forward(batch)
            Model-->>Agent: predictions [batch, 3, H, W]
            Agent->>Agent: argmax(predictions, dim=1)
            Agent->>Agent: accumulate slices
        end
        
        Agent->>Agent: restore original shape
        Agent-->>Experiment: prediction_volume
        
        Experiment->>Utils: Dice3d(prediction, ground_truth)
        Utils-->>Experiment: dice_score
        
        Experiment->>Utils: Jaccard3d(prediction, ground_truth)
        Utils-->>Experiment: jaccard_score
        
        Experiment->>Experiment: log results
    end
    
    Experiment->>Experiment: compute mean metrics
    Experiment->>Experiment: save test results JSON
```

---

## 6. Sequence Diagram - DICOM Deployment

```mermaid
sequenceDiagram
    participant Orthanc as Orthanc PACS
    participant Script as inference_dcm.py
    participant Loader as DICOM Loader
    participant Pool as ThreadPoolExecutor
    participant Agent as UNetInferenceAgent
    participant Model as UNet
    participant Report as Report Generator
    
    Orthanc->>Script: Route study to /data/incoming
    Script->>Script: Find subdirectories
    Script->>Script: Filter HCropVolume directories
    
    Script->>Loader: get_series_for_inference(study_dir)
    Loader->>Loader: list all .dcm files
    
    Loader->>Pool: Create ThreadPoolExecutor
    loop for each DICOM file (parallel)
        Pool->>Pool: process_single_dicom(file_path)
        Pool->>Pool: pydicom.dcmread(file_path)
    end
    Pool-->>Loader: list of DICOM objects
    
    Loader->>Loader: filter by SeriesDescription
    Loader-->>Script: series_for_inference list
    
    Script->>Script: load_dicom_volume_as_numpy_from_list()
    Script->>Script: construct 3D volume [X, Y, Z]
    
    Script->>Agent: UNetInferenceAgent(parameter_file_path)
    Script->>Agent: single_volume_inference_unpadded(volume)
    
    Agent->>Agent: reshape volume to patch_size
    loop for each slice batch
        Agent->>Model: forward(batch)
        Model-->>Agent: predictions
    end
    Agent-->>Script: prediction_volume [X, Y, Z]
    
    Script->>Script: get_predicted_volumes(pred)
    Script->>Script: compute anterior, posterior, total volumes
    
    Script->>Report: create_report(inference, header, orig_vol, pred_vol)
    Report->>Pool: Create ThreadPoolExecutor
    loop for 3 slices (parallel)
        Pool->>Pool: process_single_slice_for_report()
        Pool->>Pool: normalize, colorize, composite
    end
    Pool-->>Report: 3 processed slice images
    Report->>Report: combine into final report
    Report-->>Script: report PIL.Image
    
    Script->>Script: save_report_as_dcm(header, report, path)
    Script->>Orthanc: storescu 127.0.0.1 4242 (DICOM C-STORE)
    
    Script->>Script: cleanup study directory
    Script-->>Orthanc: Inference complete
```

---

## 7. Class Diagram - Batch Processing

```mermaid
classDiagram
    class batch_inference_dcm {
        +find_hippocrop_studies(root_dir) List~str~
        +validate_study_directory(study_dir) Tuple
        +process_single_study(args) Dict
        +run_batch_inference(studies_root, model_path, output_dir, max_workers, send_to_orthanc, orthanc_host, orthanc_port, orthanc_aec) BatchProcessingStats
        +save_batch_summary(stats, output_path)
        +print_batch_summary(stats)
        +os_command(command)
        +main()
    }
    
    class BatchProcessingStats {
        -int total_studies
        -int successful
        -int failed
        -float start_time
        -float end_time
        -list results
        +__init__()
        +start()
        +finish()
        +add_success(study_name, result)
        +add_failure(study_name, error)
        +get_elapsed_time() float
        +get_summary() Dict
    }
    
    class ProcessPoolExecutor {
        <<external>>
        -int max_workers
        +submit(fn, *args) Future
        +map(fn, *iterables) Iterator
        +as_completed(futures) Iterator
    }
    
    class inference_dcm_functions {
        <<imported>>
        +get_series_for_inference(path) list
        +load_dicom_volume_as_numpy_from_list(dcmlist) tuple
        +get_predicted_volumes(pred) dict
        +create_report(inference, header, orig_vol, pred_vol) PIL.Image
        +save_report_as_dcm(header, report, path)
    }
    
    class UNetInferenceAgent_Batch {
        <<imported>>
        -UNet model
        -int patch_size
        -device device
        +__init__(parameter_file_path, model, device, patch_size)
        +single_volume_inference_unpadded(volume) ndarray
    }
    
    class subprocess {
        <<external>>
        +run(args, stdout, stderr, stdin, timeout) CompletedProcess
        +Popen(args, stdout, stderr, stdin) Popen
        +TimeoutExpired exception
    }
    
    class Orthanc_PACS {
        <<external PACS>>
        +receive(dcm_file) via storescu
        +store(study)
        +route(study)
    }
    
    batch_inference_dcm --> BatchProcessingStats : creates/uses
    batch_inference_dcm --> ProcessPoolExecutor : parallelizes studies
    batch_inference_dcm --> inference_dcm_functions : imports/uses
    batch_inference_dcm --> UNetInferenceAgent_Batch : uses per study
    batch_inference_dcm --> subprocess : executes storescu
    batch_inference_dcm --> Orthanc_PACS : sends reports to
    
    note for batch_inference_dcm "Main orchestrator for parallel\nbatch processing of multiple studies"
    note for BatchProcessingStats "Tracks success/failure,\ntiming, and results"
    note for ProcessPoolExecutor "Study-level parallelism\n(bypasses Python GIL)"
```

This diagram illustrates the high-throughput batch processing architecture for processing multiple studies concurrently. `batch_inference_dcm` orchestrates study-level parallelism using `ProcessPoolExecutor` (bypassing Python's GIL), while reusing `inference_dcm`functions for per-study DICOM loading and report generation. `BatchProcessingStats` tracks success/failure rates, timing, and `subprocess` handles DICOM C-STORE operations to `Orthanc_PACS` with timeout protection for robust clinical deployment.

---

## 8. Sequence Diagram - Batch Processing Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Main as batch_inference_dcm.main()
    participant Batch as run_batch_inference()
    participant Stats as BatchProcessingStats
    participant Pool as ProcessPoolExecutor
    participant Worker as process_single_study()
    participant InfDCM as inference_dcm functions
    participant Agent as UNetInferenceAgent
    participant Orthanc as Orthanc PACS
    
    User->>Main: python batch_inference_dcm.py /data/TestVolumes --send-to-orthanc
    Main->>Main: Parse arguments
    Main->>Main: Validate paths (convert to absolute)
    
    Main->>Batch: run_batch_inference(studies_root, model_path, ...)
    Batch->>Stats: create BatchProcessingStats()
    Batch->>Stats: start()
    
    Batch->>Batch: find_hippocrop_studies(studies_root)
    Note over Batch: Discovers all HCropVolume directories
    Batch->>Batch: Found N studies
    
    Batch->>Batch: Determine max_workers (CPU cores)
    Batch->>Pool: Create ProcessPoolExecutor(max_workers)
    
    Note over Batch,Pool: Prepare arguments for parallel execution
    Batch->>Batch: Create process_args list
    
    loop For each study (parallel)
        Batch->>Pool: submit(process_single_study, args)
        Pool->>Worker: Execute in separate process
        
        Worker->>Worker: print [1/N] Processing study...
        Worker->>Worker: validate_study_directory()
        
        Worker->>InfDCM: get_series_for_inference(study_dir)
        InfDCM->>InfDCM: ThreadPool: load DICOM files
        InfDCM-->>Worker: series list
        
        Worker->>InfDCM: load_dicom_volume_as_numpy_from_list(series)
        InfDCM-->>Worker: (volume, header)
        Worker->>Worker: print → Loaded X slices
        
        Worker->>Agent: UNetInferenceAgent(model_path)
        Worker->>Agent: single_volume_inference_unpadded(volume)
        Agent->>Agent: Batch inference on slices
        Agent-->>Worker: prediction_volume
        
        Worker->>InfDCM: get_predicted_volumes(pred_label)
        InfDCM-->>Worker: {anterior, posterior, total}
        Worker->>Worker: print → Volumes
        
        Worker->>InfDCM: create_report(volumes, header, volume, pred)
        InfDCM->>InfDCM: ThreadPool: process 3 slices
        InfDCM-->>Worker: report PIL.Image
        
        Worker->>InfDCM: save_report_as_dcm(header, report, path)
        InfDCM-->>Worker: Report saved
        Worker->>Worker: print → Report saved: path
        
        alt Send to Orthanc enabled
            Worker->>Worker: print → Sending to Orthanc...
            Worker->>Worker: subprocess.run([storescu, host, port, report])
            Worker->>Orthanc: DICOM C-STORE (with 30s timeout)
            
            alt Success
                Orthanc-->>Worker: ACK
                Worker->>Worker: print ✓ Report sent successfully
                Worker->>Worker: result['sent_to_orthanc'] = True
            else Timeout
                Worker->>Worker: print ⚠ Warning: Timeout
                Worker->>Worker: result['sent_to_orthanc'] = False
            else storescu not found
                Worker->>Worker: print ⚠ Warning: Install DCMTK
                Worker->>Worker: result['sent_to_orthanc'] = False
            else Other error
                Worker->>Worker: print ⚠ Warning: Failed
                Worker->>Worker: result['sent_to_orthanc'] = False
            end
        end
        
        Worker->>Worker: print ✓ Completed in X.XXs
        Worker-->>Pool: return result dict
        
        Pool-->>Batch: Future completed
        
        alt Study succeeded
            Batch->>Stats: add_success(study_name, result)
        else Study failed
            Batch->>Stats: add_failure(study_name, error)
        end
    end
    
    Note over Batch,Pool: Wait for all futures to complete
    
    Batch->>Stats: finish()
    Batch-->>Main: BatchProcessingStats
    
    Main->>Main: print_batch_summary(stats)
    Note over Main: Display summary table
    
    alt --save-summary flag
        Main->>Main: save_batch_summary(stats, json_path)
        Main->>Main: JSON file created
    end
    
    Main-->>User: Exit (code 0 if all successful, 1 if any failed)
```

---

## 9. Activity Diagram - Batch Processing Flow

```mermaid
flowchart TD
    Start([User runs batch_inference_dcm.py]) --> Parse[Parse CLI Arguments]
    Parse --> ValidatePaths{Validate Paths}
    ValidatePaths -->|Invalid| ErrorExit[Print Error & Exit]
    ValidatePaths -->|Valid| Convert[Convert to Absolute Paths]
    
    Convert --> Search[Search for HCropVolume Studies]
    Search --> CheckStudies{Studies Found?}
    CheckStudies -->|No| ErrorExit
    CheckStudies -->|Yes| InitStats[Initialize BatchProcessingStats]
    
    InitStats --> DetectWorkers[Determine max_workers<br/>min CPU cores, study count]
    DetectWorkers --> CreatePool[Create ProcessPoolExecutor]
    CreatePool --> PrepArgs[Prepare Arguments for Each Study]
    
    PrepArgs --> SubmitJobs[Submit All Studies to Pool]
    
    SubmitJobs --> ParallelBlock[Process Studies in Parallel]
    
    subgraph ParallelBlock [Parallel Execution per Study]
        direction TB
        StartStudy[Start Study Processing] --> ValidateStudy{Valid Study?}
        ValidateStudy -->|No| StudyFail[Record Failure]
        ValidateStudy -->|Yes| LoadSeries[Get Series for Inference]
        
        LoadSeries --> LoadVolume[Load DICOM Volume<br/>ThreadPoolExecutor]
        LoadVolume --> LoadModel[Load UNet Model]
        LoadModel --> RunInference[Run Inference on Volume]
        
        RunInference --> CalcVolumes[Calculate Hippocampus Volumes]
        CalcVolumes --> CreateReport[Create Visual Report<br/>ThreadPoolExecutor]
        CreateReport --> SaveReport[Save Report as DICOM]
        
        SaveReport --> CheckOrthanc{Send to Orthanc?}
        CheckOrthanc -->|No| StudySuccess[Record Success]
        CheckOrthanc -->|Yes| CheckStorescu{storescu Available?}
        
        CheckStorescu -->|No| WarnNoDCMTK[⚠ Warning: Install DCMTK]
        WarnNoDCMTK --> StudySuccess
        
        CheckStorescu -->|Yes| SendDICOM[subprocess.run storescu<br/>30s timeout]
        SendDICOM --> CheckSend{Send Successful?}
        
        CheckSend -->|Success ✓| StudySuccess
        CheckSend -->|Timeout| WarnTimeout[⚠ Warning: Timeout]
        WarnTimeout --> StudySuccess
        CheckSend -->|Error| WarnError[⚠ Warning: Failed]
        WarnError --> StudySuccess
    end
    
    ParallelBlock --> WaitComplete[Wait for All Studies to Complete]
    WaitComplete --> CalculateStats[Calculate Statistics]
    
    CalculateStats --> PrintSummary[Print Summary Table]
    PrintSummary --> CheckSaveJSON{--save-summary?}
    
    CheckSaveJSON -->|Yes| SaveJSON[Save JSON Summary]
    CheckSaveJSON -->|No| CheckFailures
    SaveJSON --> CheckFailures{Any Failures?}
    
    CheckFailures -->|Yes| Exit1[Exit Code 1]
    CheckFailures -->|No| Exit0[Exit Code 0]
    
    Exit0 --> End([Complete])
    Exit1 --> End
    ErrorExit --> End
    
    %% Parallel Execution Block - Black background with white border
    style ParallelBlock fill:#1a1a1a,stroke:#fff,stroke-width:4px,color:#fff
    
    %% Inside Parallel Block - Black and White theme
    style StartStudy fill:#fff,stroke:#000,stroke-width:3px,color:#000
    style LoadSeries fill:#000,stroke:#fff,stroke-width:3px,color:#fff
    style LoadVolume fill:#fff,stroke:#000,stroke-width:3px,color:#000
    style LoadModel fill:#000,stroke:#fff,stroke-width:3px,color:#fff
    style RunInference fill:#fff,stroke:#000,stroke-width:3px,color:#000
    style CalcVolumes fill:#000,stroke:#fff,stroke-width:3px,color:#fff
    style CreateReport fill:#fff,stroke:#000,stroke-width:3px,color:#000
    style SaveReport fill:#000,stroke:#fff,stroke-width:3px,color:#fff
    style SendDICOM fill:#fff,stroke:#000,stroke-width:3px,color:#000
    style WarnNoDCMTK fill:#666,stroke:#fff,stroke-width:3px,color:#fff
    style WarnTimeout fill:#666,stroke:#fff,stroke-width:3px,color:#fff
    style WarnError fill:#666,stroke:#fff,stroke-width:3px,color:#fff
    style StudySuccess fill:#fff,stroke:#000,stroke-width:4px,color:#000
    style StudyFail fill:#000,stroke:#fff,stroke-width:4px,color:#fff
    
    %% Outside Parallel Block - Keep original light colors
    style Exit0 fill:#4CAF50,color:#fff
    style Exit1 fill:#F44336,color:#fff
```

---

## 10. State Diagram - Study Processing States

```mermaid
stateDiagram-v2
    [*] --> Discovered: Study found in directory
    
    Discovered --> Validating: Begin validation
    Validating --> Invalid: No DICOM files found
    Validating --> Valid: DICOM files confirmed
    Invalid --> Failed: Record error
    
    Valid --> LoadingSeries: Get series for inference
    LoadingSeries --> LoadingVolume: Series retrieved
    LoadingVolume --> LoadingModel: Volume constructed
    LoadingModel --> RunningInference: Model loaded
    
    RunningInference --> CalculatingVolumes: Predictions complete
    CalculatingVolumes --> CreatingReport: Volumes computed
    CreatingReport --> SavingReport: Report generated
    SavingReport --> CheckOrthanc: Report saved
    
    CheckOrthanc --> SendingToOrthanc: --send-to-orthanc enabled
    CheckOrthanc --> Success: Orthanc disabled
    
    SendingToOrthanc --> CheckingStorescu: Attempting upload
    CheckingStorescu --> StorescuMissing: Command not found
    CheckingStorescu --> Uploading: storescu available
    
    Uploading --> UploadSuccess: C-STORE ACK received
    Uploading --> UploadTimeout: 30s timeout exceeded
    Uploading --> UploadError: Connection/other error
    
    StorescuMissing --> SuccessWithWarning: Continue (non-fatal)
    UploadTimeout --> SuccessWithWarning: Continue (non-fatal)
    UploadError --> SuccessWithWarning: Continue (non-fatal)
    UploadSuccess --> Success: All complete
    
    Success --> [*]: result['success'] = True
    SuccessWithWarning --> [*]: result['success'] = True, warnings logged
    Failed --> [*]: result['success'] = False
    
    note right of CheckOrthanc
        Orthanc integration is optional
        Failures are non-fatal warnings
    end note
    
    note right of Uploading
        subprocess.run with:
        - 30 second timeout
        - DEVNULL stdio
        - Non-blocking execution
    end note
```

---

**Validation Plan:** For detailed clinical validation procedures and FDA submission guidelines, see [VALIDATION_PLAN.md](section3/out/VALIDATION_PLAN.md).

## Demo on CPU Hardware

### Study Ingestion (Single Study)
<img src="./section3/out/demo1.gif" width="800"/>

*Single study processing with inference_dcm.py showing real-time progress through DICOM loading, volume construction, inference, and report generation.*

### Batch Study Processing
<img src="./section3/out/demo2.gif" width="800"/>

*Parallel batch processing with batch_inference_dcm.py demonstrating concurrent processing of multiple studies with aggregate statistics and performance metrics.*
