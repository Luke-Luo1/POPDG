# Dataset Information
PopDanceSet has undergone a new upgrade. It has phased out some lower-quality dance clips while adding new popular dance videos. The selection period is not limited to the one-year range mentioned in the paper. The basic information of the new dataset is as follows:
| Dataset | Number_of_Music  | Train Seconds | Test Seconds |
| ------------- | --------------- | --------------- |--------------- |
| PopDanceSet   | 237   | 12713  | 355 |

In fact, the training set in the AIST++ dataset only totals 13,378 seconds, so from a training perspective, PopDanceSet is essentially on par with AIST++.
In the upgraded version of the database, there are a total of 305 videos, which are divided into 1,036 video clips. The distribution of video clips by duration is as follows:
| Duration | <12s  | 12-29.5s | >29.5s |
| ------------- | --------------- | --------------- |--------------- |
| Number of Clips   | 493   | 529  | 14 |

Compared to the AIST++ dataset, where 85% of dance video durations are shorter than 12 seconds, the dance clips in PopDanceSet are significantly longer.

# Results
Four methods have also yielded new results on the upgraded version of PopDanceSet:

| Method  | PFC ↓  | PBC →  | Div_k ↑ | Div_g ↑ | Beat Align Scores ↑ |
|---------|--------|--------|---------|---------|---------------------|
| GroundTruth| 1.5824         | 8.7365 | 9.0219 | 7.2931 | 0.174      |
| FACT       | 5.8791         | 1.0200 | 6.8245 | 4.3387 | 0.206      |
| Bailando   | 3.9751         | 4.8863 | 5.1835 | 5.4342 | 0.230      |
| EDGE       | 3.8366         | 4.0348 | 6.1709 | 5.7568 | 0.224      |
| POPDG      | 1.8253         | 5.9492 | 7.1342 | 5.8314 | 0.233      |

Although there are inevitable discrepancies between the specific experimental results and those reported in the paper, the verification of model performance is fundamentally consistent with the published findings. 
- Note：In terms of experimental effectiveness, iDDPM generates notably smoother dance movements compared to other generative frameworks when dealing with data of compromised quality (since the dance pose data in PopDanceSet are derived solely from monocular pose estimation algorithms coupled with manual filtering, which inherently lacks the quality of data extracted using motion capture device and multi-position cameras). This is particularly beneficial in scenarios where high-end equipment is lacking, and also enhances the tolerance for data quality variations.

# Dataset Download
Please visit [here](https://drive.google.com/file/d/11phw8Xxcnx5h4yYQVpLqDdKWSkHpBZuu/view?usp=sharing) to download and unzip the PopDanceSet in './data/' folder. Then we could preprocess the dataset using:
```
python create_dataset.py --extract-baseline --extract-jukebox
```
