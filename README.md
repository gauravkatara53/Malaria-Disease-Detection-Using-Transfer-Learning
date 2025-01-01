# Malaria-Disease-Detection-Using-Transfer-Learning
The GitHub repository presents an end-to-end case study on Malaria Disease Detection using CNN and Transfer Learning. The goal is to predict whether a given cell image is parasitized or uninfected.

<h1>Malaria Disease Detection using CNN and Transfer Learning</h1>


![image](https://github.com/princebari/Malaria-Disease-Detection-Using-Transfer-Learning/assets/115543070/117e7863-2f5a-4421-9366-8d3b059a1ed7)


<h2> Description </h2>
Malaria is a life-threatening disease caused by the Plasmodium parasite and is a significant global health concern. Early and accurate detection of malaria-infected cells plays a crucial role in effective treatment and control of the disease. Deep learning convolutional neural networks (CNNs) combined with transfer learning techniques have emerged as powerful tools for malaria detection.

In this project, a deep learning CNN model is employed for malaria disease detection. The CNN architecture is designed to learn and extract relevant features from cell images that can discriminate between infected (parasitized) and uninfected cells. By training the model on a large dataset of labeled cell images, it can learn to classify new, unseen images accurately.

Transfer learning is utilized to improve the performance of the CNN model. Pre-trained models, such as InceptionV3 or ResNet, trained on massive datasets like ImageNet, are used as a starting point. The pre-trained model's weights are frozen, and only the final layers are fine-tuned using the malaria cell image dataset. This approach allows the model to leverage the learned features from the pre-trained model, enabling it to achieve better accuracy and faster convergence.

The malaria disease detection system takes an input image of a cell and passes it through the trained CNN model. The model makes predictions on whether the cell is infected (parasitized) or uninfected. The output provides valuable information for healthcare professionals to diagnose and treat malaria effectively.

<h2>Motivation</h2>
The motivation behind the Malaria Disease Detection case study using deep learning CNN and transfer learning techniques is to develop an automated and efficient system for accurately identifying malaria-infected cells. By leveraging the power of machine learning, we aim to improve diagnostic processes, especially in resource-limited areas, and enable early detection and prompt treatment of malaria. This case study seeks to contribute to global health efforts by utilizing state-of-the-art technologies to combat a widespread infectious disease, ultimately improving patient outcomes and saving lives.

<h2>Business Objective</h2>
The objective of this case study is to develop an automated malaria disease detection system using deep learning techniques. The current diagnostic process for malaria relies heavily on manual examination of blood smears, which is time-consuming, labor-intensive, and prone to human error. By leveraging the power of deep learning algorithms, the goal is to create a robust and efficient solution that can accurately identify malaria-infected cells from microscopic images, enabling early and accurate diagnosis of the disease.


  <h2>Dataset URL</h2>

Refer :  https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria


<h2>Perfomance Metric </h2>

Accuracy
Precision and Recall
Binary Confusion Matrix

<h2>Objective</h2>
 Given a cell image Predict whether the cell is parasitized or uninfected.
 
 <h2>Results</h2>


