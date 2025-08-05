# DataDoP

## Data Availability Statement
We are committed to maintaining transparency and compliance in our data collection and sharing methods. Please note the following:

- **Publicly Available Data**: The data utilized in our studies is publicly available. We do not use any exclusive or private data sources.

- **Data Sharing Policy**: Our data sharing policy aligns with precedents set by prior works, such as [InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid), [Panda-70M](https://snap-research.github.io/Panda-70M/) 
, and [Miradata](https://github.com/mira-space/MiraData). Rather than providing the original raw data, we only supply the YouTube video IDs necessary for downloading the respective content.

- **Usage Rights**: The data released is intended exclusively for research purposes. Any potential commercial usage is not sanctioned under this agreement.

- **Compliance with YouTube Policies**: Our data collection and release practices strictly adhere to YouTube’s data privacy policies and fair of use policies. We ensure that no user data or privacy rights are violated during the process.

- **Data License**: The dataset is made available under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

### Clarifications
- The DataDoP dataset is only available for informational purposes only. The copyright remains with the original owners of the video.
- All videos of the DataDoP dataset are obtained from the Internet which is not the property of our institutions. Our institution is not responsible for the content or the meaning of these videos.
- You agree not to reproduce, duplicate, copy, sell, trade, resell, or exploit for any commercial purposes, any portion of the videos, and any portion of derived data. You agree not to further copy, publish, or distribute any portion of the DataDoP dataset.

## Dataset Construction Pipeline
### Dataset Overview
The `DataDoP` dataset comprises 29K video clips curated from online artistic videos. Each data sample includes metadata such as ClipID, YouTubeID, StartTime, EndTime, CropSize. In addition to the raw data, the processed dataset features captions, the RGBD of the first frame, and extracted camera trajectories. These trajectories have been subsequently cleaned, smoothed, and interpolated into fixed-length sequences.

### Dataset Format
The [`dataset/metadata.csv`](metadata.csv) file contains the following columns:
- **ClipID**: The Video ID for the video and its corresponding shot ID, formatted as `1_0000/shot_0014`.
- **YouTubeID**: The YouTube ID of the original video (e.g., `dfo_rMmdi0A`). The source video URL can be found at `https://www.youtube.com/watch?v={youtubeid}`.
- **StartTime**: The start time of the video segment in seconds.
- **EndTime**: The end time of the video segment in seconds.
- **CropSize**: The cropping parameters in the format used by `ffmpeg`, typically formatted as `w:h:x:y` (e.g., `640:360:0:30`).

The Dataset format is as follows:
```bash
DataDoP // DataDoP Dataset
├── <VideoID> 
│   ├── <ShotID>_caption.json
│   │       // Contains the caption text describing the shot.
│   │       // Includes:
│   │       //   - Movement (Motion Caption)
│   │       //   - Detailed Interaction
│   │       //   - Concise Interaction (Directorial Caption)
│   ├── <ShotID>_rgb.png
│   │       // RGB image (initial frame) from the shot, stored as a PNG
│   ├── <ShotID>_depth.npy
│   │       // Depth map (initial frame) from the shot, stored in NumPy .npy format
│   ├── <ShotID>_intrinsics.txt
│   │       // Camera intrinsics from Monst3R
│   ├── <ShotID>_traj.txt
│   │       // Camera extrinsics from Monst3R
│   ├── <ShotID>_transforms_cleaning.json
│   │       // Cleaned, smoothed, and interpolated camera trajectory data (in fixed-length format)
```

### Data Collection and Filtering 
- **Sources**: Shots with VideoIDs starting with `0_` are from MovieNet, where the VideoID remains the same as the original. Shots with VideoIDs starting with `1_` were sourced from YouTube, focusing on artistic videos such as movies, series, and documentaries.

**Example Data Entry**
| VideoID | YouTubeID | StartTime | EndTime | CropSize |
|---------|-----------|-----------|---------|----------|
|  |  |  |  |  |


### Data Processing Pipeline
Here are the instructions for running the data processing scripts to reproduce the DataDoP dataset.
1. Remove Black Borders from Videos:
```bash
```
2. Video Splitting:
```bash
```
3. Image Extraction:
```bash
```
4. Filter Out Static, Too Dark, or Poor Tracking Shots:
```bash
```
5. Monst3r for Camera Trajectories:
```bash
```
6. Dataset Building:
```bash
```
7. Pose Checking:
```bash
```
8. Pose Cleaning:
```bash
```
9. Trajectory Visualization:
```bash
```
10. Tagging and Pose Annotation:
```bash
```
11. Image Stitching:
```bash
```
12. Generate Interaction Captions:
```bash
```
13. Generate Captions:
```bash
```

## License
The `DataDoP` dataset is available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Please ensure proper attribution when using the dataset in research or other projects.

## Citation
If you use `DataDoP` in your research, please cite it as follows:

```markdown
@misc{zhang2025gendopautoregressivecameratrajectory,
      title={GenDoP: Auto-regressive Camera Trajectory Generation as a Director of Photography}, 
      author={Mengchen Zhang and Tong Wu and Jing Tan and Ziwei Liu and Gordon Wetzstein and Dahua Lin},
      year={2025},
      eprint={2504.07083},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.07083}, 
}
```