# Hybrid CNN-ViT Model for Hand Gesture Recognition

This project develops a fast, accurate, and generalizable system for hand gesture recognition by combining Vision Transformers (ViTs) with CNNs.

## Problem Statement

Hand Gesture Recognition (HGR) is crucial for advancing human-computer interactions in areas like AR/VR, robotic control, and assistive systems[cite: 3]. Developing a robust HGR system remains an open research challenge due to variability in hand characteristics, lighting, camera angles, and dynamic backgrounds[cite: 5]. These factors can reduce model performance and lead to overfitting, especially when training data lacks diversity or models are overfitted to specific scenarios[cite: 6]. Traditional deep learning architectures often struggle to capture fine-grained hand articulations and dynamics necessary for distinguishing subtle gestures, particularly with occlusions or in low-light settings[cite: 7, 8].

## Project Goal

The goal was to address the limitations of single models by combining ViTs with CNNs[cite: 10]. The project introduces a tailored CNN/ViT hybrid architecture that leverages CNN's local pattern sensitivity and ViT's spatial reasoning capabilities[cite: 12]. The performance of this combined architecture was studied and compared with standalone CNN and ViT models[cite: 11].

## Tech Stack

The project was developed and implemented using Python 3.10 with the following libraries[cite: 13]:

* **NumPy**: For numerical and matrix operations.
* **PyTorch**: For model development, training, and tensor operations.
* **Matplotlib**: To visualize accuracy and loss graphs.
* **Seaborn**: For enhanced visualizations like the confusion matrix.
* **Scikit-learn**: To compute evaluation metrics like accuracy and F1-score.
* **ViT (Vision Transformer for Classification)**: Custom implementation for transformer-based image classification.

## Dataset

The project used the HaGRID (Hand Gesture Recognition Image Dataset)[cite: 14].

* Large Image Source dataset for Hand Gesture Recognition[cite: 14].
* Used in Video Conferencing services (Zoom, Skype, Discord,.), Home Automation Systems, Automotive sector etc.,[cite: 14].
* Contains 552,992 full HD (1920 X 1080) RGB images into 18 classes[cite: 15].
* It contains 34,730 unique people[cite: 15].
* Pictures are mostly taken indoors and extreme conditions such as facing, backing to a window[cite: 16].
* Annotations consist of bounding boxes of hands with gesture labels in COCO format[cite: 16].
* Dataset Link: [https://huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p](https://huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p)

Exploratory Data Analysis (EDA) was performed to understand the pattern and relationship of data, its distribution, outliers etc.,[cite: 18]. This included analyzing Class Distribution[cite: 19], Top 10 most frequent Gestures (Classes)[cite: 20], Dimensionality Reduction using t-SNE[cite: 22], Aspect Ratio Distribution[cite: 42], and Preprocessing Parameter Selection: Bounding Box Padding[cite: 34].

The t-SNE visualization showed that while certain gesture classes exhibited clustering, a significant degree of overlap was observed among several classes, indicating that their hand crop features share substantial visual similarity in the pixel space[cite: 27, 28, 29]. This implied that the boundaries between these gestures are not linearly separable using simple features [cite: 30] and highlighted the necessity of utilizing more expressive models such as CNNs[cite: 31, 32].

An analysis of bounding box padding was conducted to ensure hand crops effectively captured the entire gesture[cite: 34]. Visual comparisons showed that a padding of 5 pixels provided the best trade-off, generally ensuring the entire hand was contained within the crop while adding only minimal surrounding background or wrist area[cite: 39, 40, 41].

## Model Architecture

* **CNN:** We utilized the ResNet50 model available in torchvision for the standalone CNN architecture[cite: 45]. The pre-trained weights using IMAGENET1K\_V1 were used to initialize the model[cite: 46]. The final layer was replaced to match the 18 classes for the hand gesture dataset[cite: 47]. All the layers of the ResNet50 model were allowed to train on the hand gesture dataset[cite: 49].

* **ViT:** We utilized the vit\_b\_32 model available in torchvision for the standalone ViT architecture[cite: 51]. The pre-trained weights using IMAGENET1K\_V1 were used to initialize the mode[cite: 52]. The final encoder layer of the vit\_b\_32 model was replaced to match the 18 classes[cite: 53]. For the ViT model we only allowed the last 6 layers of the encoder were allowed to train on the hand gesture dataset[cite: 55]. All other layers of the ViT were kept fixed with the pre-trained weights[cite: 56].

* **Hybrid:** A CNN model, ResNet50, serves as the backbone of the hybrid architecture[cite: 58]. Features are extracted from CNN before they are passed to the fully connected layer[cite: 59]. The feature maps are re-shaped to match the ViT's expected input format and linearly projected onto a lower embedding-dimensional space[cite: 60, 61]. This sequence of patches is passed to the ViT for classification[cite: 61]. The ViT implementation was based on [https://github.com/tintn/vision-transformer-from-scratch](https://github.com/tintn/vision-transformer-from-scratch)[cite: 62].

## Experiment Details

Experiments were conducted to train and evaluate the performance of the CNN-only, ViT-only, and hybrid CNN-ViT models using a 150K sample size[cite: 63, 64, 66]. For each model, experiments were run using both original dataset images and images cropped using bounding box data available in the dataset[cite: 63, 64, 66].

The following training configuration was used for all experiments[cite: 68]:

* Loss Function: Cross Entropy Loss
* Optimizer: Adam
* Learning Rate: 0.001
* Batch Size: 4
* Epochs: 10
* Scheduler: ReduceLROnPlateau to reduce learning rate when validation loss plateaus (Factor=0.1, Patience=2).

## Results

* **Experiment 1: CNN-Only Classification**
    * **Without cropping:** Achieved excellent test accuracy (~95%) with stable training and validation metrics[cite: 80]. This indicates strong generalization and reliable gesture classification even without precise hand localization[cite: 81]. The confusion matrix showed most gestures were classified correctly with high precision and recall[cite: 71].
    * **With cropping:** Showed strong generalization with over 90% test accuracy and a stable learning curve[cite: 91]. The confusion matrix indicated high classification accuracy across most classes, though some confusion areas existed between similar gestures[cite: 82].

* **Experiment 2: VIT-Only Classification**
    * **Without cropping:** The ViT model learned well on the training data but did not generalize well to unseen data (validation accuracy stagnated around 74-75%)[cite: 105, 106]. This was possibly due to lack of explicit hand cropping, high visual similarity between certain gestures, and overfitting[cite: 106]. The confusion matrix revealed noticeable confusion among similar gesture classes[cite: 96].
    * **With cropping:** Achieved a test accuracy above 90% (90.66%)[cite: 112], which is a notable improvement over models trained on uncropped data[cite: 117]. Using cropped hand gestures significantly improved model focus and classification accuracy[cite: 117]. The confusion matrix showed the model performed exceptionally well across most classes[cite: 108].

* **Experiment 3: Hybrid CNN-VIT Model**
    * **Without cropping:** Showed robust classification with relatively few misclassifications across gesture categories, affirming the strength of this multi-modal fusion approach (Test Accuracy 93.23%)[cite: 125, 126]. Most gesture classes were classified with high accuracy[cite: 123].
    * **With cropping:** Demonstrated robust performance, achieving over 90% test accuracy (90.85%) with well-aligned training and validation metrics[cite: 144]. It effectively balances spatial (CNN) and contextual (ViT) cues, making it suitable for real-world hand gesture recognition tasks[cite: 144]. The confusion matrix showed the model performs well across most gesture classes[cite: 134].

From the experiments, the Fusion\_CNN-ViT model generally outperformed by delivering higher accuracy, better stability in validation performance, and shorter training time[cite: 131]. The fusion of spatial (CNN) and global (ViT) features helps the model understand gestures more robustly, particularly for visually similar classes[cite: 132].

Cropping of the original images to the hand gestures did not yield improved results for the CNN or the hybrid model, as both had a similar accuracy of around 90%[cite: 155]. Preprocessing of the images for cropping was beneficial for the ViT only model, which resulted in improved from 74% to 90% accuracy[cite: 156].

## Conclusion

In conclusion, there are distinct advantages and limitations between CNNs, ViTs, and hybrid CNN-VIT models[cite: 145]. CNNs were particularly effective at learning local spatial features[cite: 146], while ViTs offered a different perspective by capturing global dependencies using self-attention mechanisms[cite: 148]. However, ViTs tended to struggle on smaller datasets[cite: 150]. The most notable insight came from working with hybrid CNN-VIT architectures, which leveraged the local feature extraction capabilities of CNNs and combined them with the global attention mechanisms of ViTs[cite: 151, 152]. The architecture that used early feature fusion showed better overall performance[cite: 153]. While ViTs alone might not be ideal for limited datasets, combining them with CNNs allows for a balanced approach—maintaining both accuracy and efficiency[cite: 157]. This hybrid strategy seems particularly promising for real-world applications with moderate data availability[cite: 158].

## Research Directions

Based on this project, a few research directions include[cite: 159]:

* **Domain-Specific Fine-Tuning:** Fine-tuning ViTs on large, domain-relevant datasets could enhance performance and applicability in real-world tasks[cite: 159].
* **Explore Dynamic Feature Fusion:** Incorporating attention-based fusion between CNN and ViT layers may lead to better integration of spatial and global features, improving prediction accuracy[cite: 160].
* **Use of Self-Supervised Learning:** Leveraging unlabeled data through self-supervised methods could improve model generalization, especially in low-data settings[cite: 161].

## Research Paper References

1.  A Methodological and Structural Review of Hand Gesture Recognition Across Diverse Data Modalities - [https://arxiv.org/abs/2408.05436](https://arxiv.org/abs/2408.05436) [cite: 162]
2.  Real-Time Hand Gesture Recognition: Integrating Skeleton-Based Data Fusion and Multi-Stream CNN - [https://arxiv.org/abs/2406.15003](https://arxiv.org/abs/2406.15003) [cite: 162]
3.  An Advanced Deep Learning Based Three-Stream Hybrid Model for Dynamic Hand Gesture Recognition - [https://arxiv.org/abs/2408.08035](https://arxiv.org/abs/2408.08035) [cite: 162]
